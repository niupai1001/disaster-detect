"""Independent native IDW, nearest-neighbor and spectral linear baselines."""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
import torch

from encoder_v3.data import sha256
from encoder_v3.hierarchical.simple_decoder import nearest_weights
from encoder_v3.shared_mae.train import write

from .data import task_view
from .experiment import coordinates, metric_means, native_metrics, read_members


def native_xy(grid, pixels):
    dx, dy = grid.spacing
    _, _, left, _, _, top = grid.affine
    return np.column_stack((left + (pixels % grid.shape[1] + .5) * dx,
                            top - (pixels // grid.shape[1] + .5) * dy))


def sample_bands(bands, grid, pixels):
    positions = native_xy(grid, pixels)
    sources = []
    for band, visible in bands:
        dx, dy = band.grid.spacing
        _, _, left, _, _, top = band.grid.affine
        x = np.floor((positions[:, 0] - left) / dx).astype(int)
        y = np.floor((top - positions[:, 1]) / dy).astype(int)
        h, w = band.grid.shape
        inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        v = np.full(len(pixels), np.nan, np.float64)
        y, x = y[inside], x[inside]
        v[inside] = np.where(visible.numpy()[y, x], band.values.numpy()[y, x], np.nan)
        sources.append(v)
    return np.column_stack((np.ones(len(pixels)), *sources))


def fit_spectral_grams(rows, dataset, device):
    grams = {}
    for row in rows:
        observation = dataset.load(row, device)
        bands = sorted(observation.bands, key=lambda b: b.key)
        keys = tuple(b.key for b in bands)
        for band in bands:
            ids = np.flatnonzero(band.valid.numpy())
            matrix = sample_bands([(b, b.valid) for b in bands if b is not band], band.grid, ids)
            response = band.values.numpy().flatten()[ids]
            values = np.column_stack((matrix, response))
            finite = values[np.isfinite(values).all(1)]
            key = (row['sensor'], keys, band.key)
            if key not in grams:
                grams[key] = finite.T @ finite
            else:
                grams[key] += finite.T @ finite
    return grams


def evaluate_references(out, dataset, stride):
    device = torch.device('cpu')
    train, _ = read_members(out, 'TRAIN')
    val, release = read_members(out, 'VAL')
    panel = out / 'members/val_panel.npz'
    if sha256(panel) != release['panels']['VAL']['sha256']:
        raise ValueError('Reference evaluation panel changed')
    grams = fit_spectral_grams(train, dataset, device)
    observations = defaultdict(list)
    targets = defaultdict(int)
    covered = defaultdict(int)
    with np.load(panel, allow_pickle=False) as fixed:
        for row in val:
            obs = dataset.load(row, device)
            native = {b.key: b for b in obs.bands}
            for kind in ('A', 'B1'):
                view = task_view(obs, kind, stride, panel=coordinates(fixed, obs, kind))
                if view is None:
                    continue
                by_method = defaultdict(list)
                for key, request, labels in zip(view.keys, view.requests, view.truth, strict=True):
                    band = native[key]
                    ids = request.pixels.numpy()
                    truth = np.full(request.grid.shape, np.nan, np.float32)
                    truth.flat[ids] = labels.numpy()
                    predictions = {}
                    if kind == 'A':
                        observed = band.valid.numpy().flatten().copy()
                        observed[ids] = False
                        source = np.flatnonzero(observed)
                        if len(source):
                            points, query = native_xy(band.grid, source), native_xy(band.grid, ids)
                            values = band.values.numpy().flatten()[source]
                            neighbors, weight = nearest_weights(points, query)
                            predictions['idw'] = (values[neighbors] * weight).sum(1)
                            predictions['nearest'] = values[cKDTree(points).query(query, workers=1)[1]]
                        else:
                            predictions = {m: np.full(len(ids), np.nan) for m in ('idw', 'nearest')}
                    else:
                        keys = sorted(native)
                        hidden = set(view.keys)
                        other = [k for k in keys if k != key]
                        visible = [k for k in other if k not in hidden]
                        gram = grams[(obs.sensor, tuple(keys), key)]
                        columns = [0, *(i+1 for i,k in enumerate(other) if k in visible)]
                        beta = np.linalg.lstsq(gram[np.ix_(columns, columns)], gram[columns, -1], rcond=None)[0]
                        predictors = [(native[k], native[k].valid) for k in visible]
                        predictions['linear'] = sample_bands(predictors, band.grid, ids) @ beta
                    for method, values in predictions.items():
                        supported = np.isfinite(values)
                        group = (obs.sensor, kind, method)
                        targets[group] += len(ids)
                        covered[group] += int(supported.sum())
                        if supported.any():
                            prediction = np.full_like(truth, np.nan)
                            prediction.flat[ids] = values
                            by_method[method].append(native_metrics(prediction, truth))
                for method, metrics in by_method.items():
                    observations[obs.sensor, kind, method].append(metric_means(metrics))
    report = [dict(sensor=s, task=t, method=m, targets=targets[s,t,m], covered=covered[s,t,m],
                   observations=len(observations[s,t,m]),
                   metrics=metric_means(observations[s,t,m]) if observations[s,t,m] else None)
              for s,t,m in sorted(targets)]
    write(out / 'references_val.json', dict(train=release['roles']['TRAIN'], val=release['roles']['VAL'],
                                           panel=release['panels']['VAL']['sha256'], groups=report))
