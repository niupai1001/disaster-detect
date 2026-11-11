"""Exact indexed multi-head attention with bounded edge intermediates."""
from collections import defaultdict
import time

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Execution limits only. They do not change queries, edges, or precision.
PROJECT_ROWS = 262144
DEVICE_PROJECT_BYTES = 768 * 1024**2


class Metrics:
    def __init__(self):
        self.attention = defaultdict(lambda: dict(pairs=0, q=0, k=0, v=0, flops=0,
                                                   degree=defaultdict(int), pair_flops=0, index_seconds=0.,
                                                   gather_scatter_ms=0.))
        self.tokens = {}
        self.channel_degree = defaultdict(int)
        self.index_events = defaultdict(list)

    def summarize(self):
        for name, events in self.index_events.items():
            if self.attention[name]["gather_scatter_ms"] is not None:
                self.attention[name]["gather_scatter_ms"] = sum(a.elapsed_time(b) for a, b in events)
        return dict(tokens=self.tokens, attention=dict(self.attention),
                    cross_channel_degree=dict(self.channel_degree))


def mark(device, events):
    if events is None or device.type != "cuda":
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device))
    return event


def finish(device, events, begin):
    if begin is not None:
        events.append((begin, mark(device, events)))


def edge_terms(q, k, v, src, dst, scale, events=None):
    device = q.device
    begin = mark(device, events)
    query, key, value = q[src], k[dst], v[dst]
    finish(device, events, begin)
    score = (query * key).sum(-1).float() * scale
    index = src[:, None].expand_as(score)
    begin = mark(device, events)
    largest = score.new_full(q.shape[:2], -torch.inf)
    largest.scatter_reduce_(0, index, score, reduce="amax", include_self=True)
    maximum = largest[src]
    finish(device, events, begin)
    weight = (score-maximum).exp()
    begin = mark(device, events)
    denominator = score.new_zeros(q.shape[:2]).index_add_(0, src, weight)[src]
    finish(device, events, begin)
    return query, key, value, (weight/denominator).to(value.dtype)


def attention_rows(q, k, v, src, dst, device):
    src = torch.as_tensor(src, dtype=torch.long, device=device)
    if k.device != device:
        keys, dst = np.unique(dst, return_inverse=True)
        keys = torch.as_tensor(keys, dtype=torch.long, device=k.device)
        k, v = k[keys].to(device), v[keys].to(device)
    else:
        keys = None
    dst = torch.as_tensor(dst, dtype=torch.long, device=device)
    return q.to(device), k, v, src, dst, keys


class IndexedAttention(torch.autograd.Function):
    """Recompute edge weights during backward, accumulating into shared K/V buffers."""

    @staticmethod
    def forward(ctx, q, k, v, graph, chunk, name, metrics, compute_device):
        ctx.save_for_backward(q, k, v)
        ctx.graph, ctx.chunk, ctx.device = graph, chunk, compute_device
        ctx.scale = q.shape[-1] ** -.5
        out = torch.zeros_like(q)
        present = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        record = metrics.attention[name] if metrics is not None else None
        events = metrics.index_events[name] if metrics is not None else None
        for start in range(0, len(q), chunk):
            end = min(start+chunk, len(q))
            before = time.perf_counter()
            source, destination = (graph(start, end, True) if record is not None and name == "fine_cross"
                                   else graph(start, end))
            if record is not None:
                record["pairs"] += len(source)
                pair_flops = 4*len(source)*q.shape[1]*q.shape[2]
                record["pair_flops"] += pair_flops
                record["flops"] += pair_flops
                degree, counts = np.unique(np.bincount(source, minlength=end-start), return_counts=True)
                for d, count in zip(degree, counts):
                    record["degree"][int(d)] += int(count)
            if not len(source):
                if record is not None:
                    record["index_seconds"] += time.perf_counter()-before
                continue
            a, b, c, src, dst, _ = attention_rows(q[start:end], k, v, source, destination, compute_device)
            if record is not None:
                record["index_seconds"] += time.perf_counter()-before
            _, _, value, weight = edge_terms(a, b, c, src, dst, ctx.scale, events)
            weighted = weight[..., None]*value
            begin = mark(compute_device, events)
            reduced = torch.zeros_like(a).index_add_(0, src, weighted)
            finish(compute_device, events, begin)
            out[start:end] = reduced.to(out.device)
            present[start:end][torch.as_tensor(source, device=present.device, dtype=torch.long)] = True
        ctx.mark_non_differentiable(present)
        return out, present

    @staticmethod
    def backward(ctx, grad, _):
        q, k, v = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for start in range(0, len(q), ctx.chunk):
            end = min(start+ctx.chunk, len(q))
            source, destination = ctx.graph(start, end)
            if not len(source):
                continue
            a, b, c, src, dst, keys = attention_rows(q[start:end], k, v, source, destination, ctx.device)
            query, key, value, weight = edge_terms(a, b, c, src, dst, ctx.scale)
            dy = grad[start:end].to(ctx.device)[src]
            dweight = (dy*value).sum(-1)
            mean = weight.new_zeros(a.shape[:2]).index_add_(0, src, weight*dweight)
            ds = (weight*(dweight-mean[src])*ctx.scale)[..., None]
            dq[start:end] = torch.zeros_like(a).index_add_(0, src, ds*key).to(dq.device)
            if keys is None:
                dk.index_add_(0, dst, ds*query)
                dv.index_add_(0, dst, weight[..., None]*dy)
            else:
                dk.index_add_(0, keys, torch.zeros_like(b).index_add_(0, dst, ds*query).to(dk.device))
                dv.index_add_(0, keys, torch.zeros_like(c).index_add_(0, dst, weight[..., None]*dy).to(dv.device))
        return dq, dk, dv, None, None, None, None, None


def rows_apply(fn, tensor, compute, storage):
    if tensor.device == compute == storage:
        return fn(tensor)
    def one(rows):
        return fn(rows.to(compute)).to(storage)
    return torch.cat([checkpoint(one, rows, use_reentrant=False) if torch.is_grad_enabled() else one(rows)
                      for rows in tensor.split(PROJECT_ROWS)])


def spill_device(device, tokens, width, element_size):
    return (torch.device("cpu") if device.type == "cuda"
            and tokens*width*element_size > DEVICE_PROJECT_BYTES else device)


class Read(nn.Module):
    def __init__(self, query, context, heads):
        super().__init__()
        self.width, self.heads = query, heads
        self.query_norm = nn.LayerNorm(query)
        self.q = nn.Linear(query, query)
        self.contexts = nn.ModuleList(nn.ModuleDict(dict(norm=nn.LayerNorm(width),
                                                        k=nn.Linear(width, query),
                                                        v=nn.Linear(width, query)))
                                      for width in ((context,) if isinstance(context, int) else context))
        self.output = nn.Linear(query, query, bias=False)

    def project_context(self, contexts, name, metrics=None):
        device = self.q.weight.device
        size = sum(map(len, contexts))
        kv_storage = spill_device(device, size, self.width, contexts[0].element_size())
        if metrics is not None:
            record = metrics.attention[name]
            record["k"] += size
            record["v"] += size
            record["flops"] += sum(4*len(x)*x.shape[-1]*self.width for x in contexts)
            record["spilled_kv"] = kv_storage != device
        shape = (-1, self.heads, self.width//self.heads)
        keys, values = [], []
        for x, layer in zip(contexts, self.contexts, strict=True):
            # Separate K/V calls allow bounded projection without retaining doubled-size concatenations.
            for kind, tensors in (("k", keys), ("v", values)):
                tensors.append(rows_apply(lambda a, layer=layer, kind=kind: layer[kind](layer["norm"](a)),
                                          x, device, kv_storage).reshape(shape))
        k = keys[0] if len(keys) == 1 else torch.cat(keys)
        v = values[0] if len(values) == 1 else torch.cat(values)
        return k, v

    def read_projected(self, query, k, v, graph, chunk, name, metrics=None):
        device = self.q.weight.device
        q_storage = spill_device(device, len(query), self.width, query.element_size())
        if metrics is not None:
            record = metrics.attention[name]
            record["q"] += len(query)
            record["flops"] += 4*len(query)*self.width**2
            record["spilled_query"] = q_storage != device
        shape = (-1, self.heads, self.width//self.heads)
        q = rows_apply(lambda x: self.q(self.query_norm(x)), query, device, q_storage).reshape(shape)
        if hasattr(graph, "attend") and max((len(keys) for _, keys in graph.blocks), default=0) > 64:
            result = graph.attend(q.to(device), k, v, metrics, name)
        else:
            if hasattr(graph, "expanded"):
                source, destination = graph.expanded()
                def indexed(start, end):
                    keep = (source >= start) & (source < end)
                    return source[keep]-start, destination[keep]
                graph = indexed
            result, _ = IndexedAttention.apply(q, k, v, graph, chunk, name, metrics, device)
        return query + rows_apply(self.output, result.flatten(1), device, query.device)

    def forward(self, query, context, graph, chunk, name, metrics=None):
        contexts = (context,) if isinstance(context, torch.Tensor) else context
        k, v = self.project_context(contexts, name, metrics)
        return self.read_projected(query, k, v, graph, chunk, name, metrics)


def full_graph(keys):
    def graph(start, end):
        return np.repeat(np.arange(end-start), keys), np.tile(np.arange(keys), end-start)
    return graph


def ff(width, ratio):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width*ratio),
                         nn.GELU(), nn.Linear(width*ratio, width))
