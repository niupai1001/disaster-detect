"""Tile both decoder reads and retain only scalar predictions and shared projections."""
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SharedCoarseContext:
    groups: np.ndarray


@dataclass
class SharedRows:
    inverse: np.ndarray


def select(start, end, graph, k, v, device, name, metrics):
    before = time.perf_counter()
    if isinstance(graph, SharedCoarseContext):
        unique, inverse = np.unique(graph.groups[start:end], return_inverse=True)
        rows = torch.as_tensor(unique, device=k.device, dtype=torch.long)
        # The shared branch supplies completed readout features and an empty V slot.
        return rows, (k[rows].to(device), v), SharedRows(inverse)
    if getattr(graph, "prefer_grouped", False):
        rows, prepared = graph.prepare(start, end)
    else:
        src, dst = graph(start, end)
        rows, compact = np.unique(dst, return_inverse=True)
        prepared = lambda _start, _end: (src, compact)
    rows = torch.as_tensor(rows, device=k.device, dtype=torch.long)
    # Tile-local indices are replayed during gradient recomputation.
    result = k[rows].to(device), v[rows].to(device)
    if metrics is not None:
        metrics.attention[name]["index_seconds"] += time.perf_counter()-before
    return rows, result, prepared


def tile(decoder, descriptor, geography, local, coarse, chunk, metrics, inverse):
    width = decoder.config.descriptor
    condition = torch.nn.functional.linear(descriptor, decoder.condition.weight[:, :width],
                                            decoder.condition.bias)
    # Packed targets usually occupy one channel per tile; broadcast avoids atomic
    # accumulation of thousands of gradients into the same descriptor row.
    repeated = condition.expand(len(geography), -1) if len(condition) == 1 else condition[inverse]
    q = repeated + torch.nn.functional.linear(geography, decoder.condition.weight[:, width:])
    if metrics is not None:
        record = metrics.attention["decoder_local"]
        record["condition_flops"] = record.get("condition_flops", 0) + 2*decoder.config.decoder*(
            len(descriptor)*width + len(geography)*5)
    q = decoder.local.read_projected(q, *local[0], local[1], chunk, "decoder_local", metrics)
    if isinstance(coarse[1], SharedRows):
        q = q + coarse[0][0][coarse[1].inverse]
    else:
        q = decoder.global_read.read_projected(q, *coarse[0], coarse[1], chunk, "decoder_coarse", metrics)
    return decoder.output(q).squeeze(-1)


class ReadoutTiles(torch.autograd.Function):
    """Recompute one complete tile at a time and reduce into shared gradients."""

    @staticmethod
    def forward(ctx, descriptor, lk, lv, rk, rv, decoder, query_features, local, coarse, count,
                metrics, *parameters):
        ctx.save_for_backward(descriptor, lk, lv, rk, rv, *parameters)
        ctx.decoder, ctx.query_features = decoder, query_features
        ctx.local, ctx.coarse, ctx.count = local, coarse, count
        chunk = (decoder.config.decoder_query_chunk if getattr(local, "prefer_grouped", False)
                 else decoder.config.query_chunk)
        predictions = descriptor.new_empty(count)
        for start in range(0, count, chunk):
            end = min(start+chunk, count)
            bands, positions = query_features(start, end)
            bands, inverse = np.unique(bands, return_inverse=True)
            _, fine, fine_edges = select(start, end, local, lk, lv, descriptor.device,
                                         "decoder_local", metrics)
            _, regional, regional_edges = select(start, end, coarse, rk, rv, descriptor.device,
                                                 "decoder_coarse", metrics)
            predictions[start:end] = tile(decoder, descriptor[bands], positions, (fine, fine_edges),
                                          (regional, regional_edges), chunk, metrics, inverse)
        return predictions

    @staticmethod
    def backward(ctx, gradient):
        descriptor, lk, lv, rk, rv, *params = ctx.saved_tensors
        model = ctx.decoder
        # Save hooks may restore parameter values as copies. Differentiate against
        # the original module parameters, which are the inputs of this function.
        parameters = (list(model.condition.parameters()) + list(model.output.parameters()) +
                      [p for block in (model.local, model.global_read)
                       for part in (block.query_norm, block.q, block.output) for p in part.parameters()])
        accum = [torch.zeros_like(x) for x in (descriptor, lk, lv, rk, rv)]
        param_grads = [None]*len(params)
        enabled = [(i, p) for i, p in enumerate(parameters) if p.requires_grad]
        chunk = (model.config.decoder_query_chunk if getattr(ctx.local, "prefer_grouped", False)
                 else model.config.query_chunk)
        for start in range(0, ctx.count, chunk):
            end = min(start+chunk, ctx.count)
            bands, positions = ctx.query_features(start, end)
            bands, inverse = np.unique(bands, return_inverse=True)
            bands = torch.as_tensor(bands, dtype=torch.long, device=descriptor.device)
            fine_rows, fine, fine_edges = select(start, end, ctx.local, lk, lv, descriptor.device,
                                                 "decoder_local", None)
            regional_rows, regional, regional_edges = select(start, end, ctx.coarse, rk, rv,
                                                             descriptor.device, "decoder_coarse", None)
            inputs = tuple(x.detach().requires_grad_() for x in (descriptor[bands], *fine, *regional))
            with torch.enable_grad():
                output = tile(model, inputs[0], positions, (inputs[1:3], fine_edges),
                              (inputs[3:5], regional_edges), chunk, None, inverse)
                grads = torch.autograd.grad(output, (*inputs, *(p for _, p in enabled)),
                                            gradient[start:end], allow_unused=True)
            for buffer, rows, grad in zip(accum, (bands, fine_rows, fine_rows, regional_rows, regional_rows),
                                          grads[:5]):
                if grad is not None:
                    buffer.index_add_(0, rows, grad.to(buffer.device))
            for (i, _), grad in zip(enabled, grads[5:]):
                if grad is not None:
                    param_grads[i] = grad if param_grads[i] is None else param_grads[i] + grad
        return (*accum, None, None, None, None, None, None, *param_grads)
