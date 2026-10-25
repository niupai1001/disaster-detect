"""Read-only SSL diagnostics: physical/standardized error and paired/nonpaired views."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from landslide_seg.data.optical_observation import OpticalObservation, collate_optical
from landslide_seg.models.frozen_use import FrozenUSE
from pretrain_use_independent import OUT, EXTERNAL, initialize, inputs
from audit_use_independent_data import read, sha

S2_NAMES = ('blue','green','red','rededge1','rededge2','rededge3','nir','nir08','swir16','swir22')
S2_IDS = ('B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12')
LS_NAMES = ('blue','green','red','nir08','swir16','swir22')


def view_statistics(left, right):
    if left.shape != right.shape or left.ndim != 2 or len(left) < 2:
        raise ValueError('At least two paired locations with matching embeddings required')
    similarity = F.normalize(left.double(), dim=1) @ F.normalize(right.double(), dim=1).T
    negative = similarity[torch.arange(len(left)), torch.arange(len(left)).roll(1)]
    return dict(locations=len(left), positive_cosine=float(similarity.diag().mean()),
        nonpaired_negative_cosine=float(negative.mean()), positive_minus_negative=float((similarity.diag()-negative).mean()),
        retrieval_recall={str(k):float((similarity.topk(min(k,len(left)),dim=1).indices == torch.arange(len(left))[:,None]).any(1).double().mean()) for k in (1,5,10)},
        left_mean_feature_std=float(left.double().std(0).mean()), right_mean_feature_std=float(right.double().std(0).mean()),
        negative_rule='Previous different cache in fixed validation order; other locations are also retrieval distractors')


def identity_aliases(rows):
    counts = Counter(name for row in rows for name in row['band_ids'])
    pairs = []
    for alias, canonical in zip(S2_NAMES, S2_IDS):
        if counts[alias] and counts[canonical]:
            pairs.append(dict(physical_band=canonical, observed_names=[alias,canonical],
                observations=[counts[alias],counts[canonical]],
                random_identity_sha256=[hashlib.sha256(('USE1234'+name).encode()).hexdigest() for name in (alias,canonical)]))
    return dict(distinct_input_names=len(counts), same_physical_band_different_names=pairs,
        interpretation='Current random identity hashes literal band names; aliases receive different fixed identities',
        consequence='Random-identity comparison also exposes source naming convention; it is not a pure ten-band physical-identity control')


def observation(row, statistics):
    path = EXTERNAL/row['file']
    if sha(path) != row['sha256']:
        raise ValueError('Audited cache changed')
    with np.load(path, allow_pickle=False) as arrays:
        x, valid, metadata, known = (arrays[k].copy() for k in ('x','pixel_valid','metadata','known'))
    mean, std = (np.asarray(statistics[k],dtype='float32') for k in ('mean','std'))
    return OpticalObservation(torch.from_numpy((x-mean[:,None,None])/std[:,None,None]), tuple(row['band_ids']),
        torch.from_numpy(metadata), torch.from_numpy(known), torch.ones(len(x),dtype=torch.bool),
        torch.from_numpy(valid), dict(crs=row['crs'],transform=row['transform']))


def reconstruction_errors(metrics, statistics):
    output = {}
    for sensor, values in metrics.items():
        index = {name:i for i,name in enumerate(S2_NAMES if sensor=='s2' else LS_NAMES)}
        if sensor == 's2': index.update({name:i for i,name in enumerate(S2_IDS)})
        bands = {}
        for name, value in values['reflectance'].items():
            std = statistics[sensor]['std'][index[name]]
            bands[name] = dict(reflectance_mae=value['mae'], reflectance_rmse=value['rmse'],
                standardized_mae=value['mae']/std, standardized_rmse=value['rmse']/std, valid_target_pixels=value['pixels'])
        output[sensor] = dict(selection_sample_band_balanced_standardized_mae=values['standardized_mae'], bands=bands,
            spatial_standardized_mae=values.get('standardized_spatial_mae'),
            note='Per-band errors pool valid target pixels; selection averages bands/samples and uses a different weighting')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    folder = OUT/'pretraining'/args.variant
    target = folder/'ssl_diagnostics.json'
    if target.exists(): raise FileExistsError('Preserve existing SSL diagnostics')
    acceptance, config = read(folder/'acceptance.json'), read(folder/'resolved_config.json')
    if acceptance['passed'] is not True or sha(folder/'best.pt') != acceptance['checkpoints']['best']['file_sha256']:
        raise ValueError('Accepted selected SSL weights required')
    splits, statistics, _, _ = inputs(args.variant)
    if statistics != config['statistics']: raise ValueError('Frozen statistics changed')
    diagnostic_config = dict(variant=args.variant, source_checkpoint_sha256=sha(folder/'best.pt'),
        validation_files={sensor:[r['file'] for r in rows] for sensor,rows in splits['PRETRAIN_VAL'].items()},
        views='First three actual bands versus remaining bands; full selected PRETRAIN_VAL for each sensor',
        training_updates=0, checkpoint_selection_changed=False, auditor_sha256=sha(Path(__file__)))
    config_path = folder/'ssl_diagnostic_config.json'
    if config_path.exists():
        if read(config_path) != diagnostic_config: raise ValueError('Diagnostic configuration differs')
    else: config_path.write_text(json.dumps(diagnostic_config,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    model = initialize(args.variant,args.device)
    model.use.load_state_dict(saved['use'],strict=True)
    encoder = FrozenUSE(model.use)
    digest = encoder.verify()
    if digest != acceptance['checkpoints']['best']['use_sha256']: raise ValueError('Selected USE state differs')
    del saved
    views = {}
    with torch.no_grad():
        for sensor, rows in splits['PRETRAIN_VAL'].items():
            left, right = [], []
            started = time.perf_counter()
            for start in range(0,len(rows),8):
                observations = [observation(row,statistics[sensor]) for row in rows[start:start+8]]
                for collector,indices in ((left,slice(0,3)),(right,slice(3,None))):
                    batch = collate_optical([obs.subset(obs.band_ids[indices]) for obs in observations]).to(args.device)
                    encoded = encoder(batch)
                    valid = encoded['valid'][:,None]
                    pooled = (encoded['features']*valid).sum((-1,-2))/valid.sum((-1,-2)).clamp_min(1)
                    collector.append(pooled.cpu())
            views[sensor] = view_statistics(torch.cat(left),torch.cat(right))
            views[sensor]['seconds'] = time.perf_counter()-started
            print(args.variant,sensor,len(rows),'views finished',flush=True)
    history = read(folder/'history.json')
    selected = next(row for row in history if row['step'] == acceptance['checkpoints']['best']['step'])
    proof = dict(passed=True, use_sha256=encoder.verify(), config=diagnostic_config,
        reconstruction=reconstruction_errors(selected['metrics'],statistics), cross_band_views=views,
        identity_alias_audit=identity_aliases(splits['TRAIN'].get('s2',[])),
        limitations=['Within-sensor same-location band views, not real cross-sensor paired retrieval',
                     'Different validation caches may be geographically adjacent; negatives are not guaranteed independent regions',
                     'Global pooled feature retrieval does not establish local segmentation quality',
                     'No additional labels, training, data downloads or checkpoint selection'])
    if proof['use_sha256'] != digest: raise ValueError('Diagnostic mutated frozen USE')
    target.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(args.variant,'SSL diagnostics passed',flush=True)


if __name__ == '__main__': main()
