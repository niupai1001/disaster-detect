"""Exact partitioned softmax over sparse native edges and shared latent sets."""
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import expand
from .routes import combine, expand_latents


@dataclass
class Prepared:
    source: np.ndarray
    destination: np.ndarray
    blocks: tuple
    count: int

    def expanded(self):
        rows, keys = [self.source], [self.destination]
        for query, context in self.blocks:
            rows.append(np.repeat(query, len(context)))
            keys.append(np.tile(context, len(query)))
        return np.concatenate(rows), np.concatenate(keys)

    def attend(self, q, k, v, metrics, name):
        device = q.device
        n, heads, width = q.shape
        zero = q.new_zeros(n, heads, width)
        sparse_output = zero
        sparse_logz = q.new_full((n, heads), -torch.inf)
        degree = np.bincount(self.source, minlength=n)
        if len(self.source):
            src = torch.as_tensor(self.source, device=device, dtype=torch.long)
            dst = torch.as_tensor(self.destination, device=device, dtype=torch.long)
            scores = (q[src]*k[dst]).sum(-1)/width**.5
            maximum = q.new_full((n, heads), -torch.inf)
            maximum.scatter_reduce_(0, src[:, None].expand_as(scores), scores.detach(), reduce="amax")
            weights = (scores-maximum[src]).exp()
            denom = q.new_zeros(n, heads).index_add(0, src, weights)
            sparse_output = zero.index_add(0, src, weights[..., None]*v[dst])/denom.clamp_min(1e-30)[..., None]
            sparse_logz = maximum + denom.clamp_min(1e-30).log()
        result = sparse_output
        pairs = len(self.source)
        buckets = {}
        for rows, keys in self.blocks:
            if len(keys):
                capacity = ((len(rows)+15)//16)*16
                buckets.setdefault((len(keys), capacity), []).append((rows, keys))
                degree[rows] += len(keys)
                pairs += len(rows)*len(keys)
        padded_pairs = 0
        for (key_count, capacity), blocks in buckets.items():
            query_index = np.zeros((len(blocks), capacity), dtype=np.int64)
            present = np.zeros_like(query_index, dtype=bool)
            for i, (rows, _) in enumerate(blocks):
                query_index[i, :len(rows)] = rows
                present[i, :len(rows)] = True
            query_index = torch.as_tensor(query_index, device=device)
            context = torch.as_tensor(np.stack([keys for _, keys in blocks]), device=device)
            present = torch.as_tensor(present, device=device)
            scores = (q[query_index].permute(0,2,1,3) @
                      k[context].permute(0,2,3,1))/width**.5
            logz = scores.logsumexp(-1).transpose(1,2)[present]
            dense = (scores.softmax(-1) @ v[context].permute(0,2,1,3)).transpose(1,2)[present]
            query = query_index[present]
            total = torch.logaddexp(sparse_logz[query], logz)
            value = (sparse_output[query]*(sparse_logz[query]-total).exp()[..., None] +
                     dense*(logz-total).exp()[..., None])
            result = result.index_copy(0, query, value)
            padded_pairs += len(blocks)*capacity*key_count
        if metrics is not None:
            record = metrics.attention[name]
            record["pairs"] += pairs
            record["executed_dense_pairs_with_padding"] = record.get("executed_dense_pairs_with_padding", 0) + padded_pairs
            record["gather_scatter_ms"] = None
            record["padding_flops"] = record.get("padding_flops", 0) + 4*heads*width*(
                padded_pairs - (pairs-len(self.source)))
            record["pair_flops"] += 4*pairs*heads*width
            record["flops"] += 4*pairs*heads*width
            unique, counts = np.unique(degree, return_counts=True)
            for d, c in zip(unique, counts):
                record["degree"][int(d)] += int(c)
            record["stored_route_indices"] = record.get("stored_route_indices", 0) + (
                len(self.source)*2 + sum(len(rows)+len(keys) for rows, keys in self.blocks))
        return result


class Factorized:
    def __init__(self, sparse, groups, group_keys, offset, latents, globals=0):
        self.sparse, self.groups, self.group_keys = sparse, groups, group_keys
        self.offset, self.latents, self.globals = offset, latents, globals
        self.prefer_grouped = (group_keys is not None and
                               int(np.diff(group_keys.offsets).max(initial=0))*latents > 64)

    def prepare(self, start, end):
        source, destination = self.sparse(start, end)
        rows, cells = self.groups(start, end)
        n = end-start
        counts = np.bincount(rows, minlength=n)
        matrix = np.full((n, max(1, int(counts.max()))), -1, dtype=np.int32)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        _, slot = expand(counts)
        matrix[sorted_rows, slot] = cells[order]
        matrix.sort(axis=1)
        signatures, inverse = np.unique(matrix, axis=0, return_inverse=True)
        blocks = []
        all_keys = [destination]
        for i, signature in enumerate(signatures):
            cells = signature[signature >= 0]
            if self.group_keys is not None:
                _, cells = self.group_keys.take(cells)
            keys = np.concatenate((np.arange(self.globals),
                                   self.offset+expand_latents(cells, self.latents)))
            queries = np.flatnonzero(inverse == i)
            if len(keys):
                blocks.append((queries, keys))
                all_keys.append(keys)
        keys = np.unique(np.concatenate(all_keys))
        sparse_keys = np.searchsorted(keys, destination)
        blocks = tuple((rows, np.searchsorted(keys, context)) for rows, context in blocks)
        return keys, Prepared(source, sparse_keys, blocks, n)

    def __call__(self, start, end):
        def expanded(lo, hi):
            rows, cells = self.groups(lo, hi)
            if self.group_keys is None:
                return rows, cells
            query, keys = self.group_keys.take(cells)
            return rows[query], keys
        return combine(self.sparse, expanded, self.offset, self.latents, self.globals)(start, end)
