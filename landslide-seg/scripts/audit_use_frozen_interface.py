"""Check trained frozen USE interfaces on existing unlabeled cache observations."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from landslide_seg.data.optical_observation import OpticalObservation, collate_optical
from landslide_seg.models.frozen_use import FrozenUSE, state_sha256
from pretrain_use_independent import OUT, EXTERNAL, initialize, inputs
from audit_use_independent_data import read, sha


def check_interface(frozen, observations, device):
    checks = []
    reference = frozen.verify()

    def run(batch):
        output = frozen(batch.to(device))
        shape = batch['x'].shape
        if output['features'].shape != (shape[0], 32, *shape[-2:]):
            raise ValueError('Feature grid or channel count changed')
        if not torch.isfinite(output['features']).all():
            raise ValueError('Nonfinite frozen USE output')
        assert not output['features'].requires_grad
        assert frozen.verify() == reference
        return output

    with torch.no_grad():
        individual = []
        for name, observation in observations:
            original = run(collate_optical([observation]))
            reversed_obs = observation.subset(observation.band_ids[::-1])
            reverse = run(collate_optical([reversed_obs]))
            torch.testing.assert_close(reverse['features'], original['features'], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(reverse['valid'], original['valid'], atol=0, rtol=0)
            # A physically absent channel must stay absent even when its values are nonfinite.
            padded = replace(observation,
                x=torch.cat((observation.x, torch.full_like(observation.x[:1], float('nan')))),
                band_ids=observation.band_ids+('absent_padding',),
                band_metadata=torch.cat((observation.band_metadata, torch.zeros_like(observation.band_metadata[:1]))),
                known_mask=torch.cat((observation.known_mask, torch.zeros_like(observation.known_mask[:1]))),
                band_valid=torch.cat((observation.band_valid, torch.zeros_like(observation.band_valid[:1]))),
                pixel_valid=torch.cat((observation.pixel_valid, torch.ones_like(observation.pixel_valid[:1]))))
            padding = run(collate_optical([padded]))
            torch.testing.assert_close(padding['features'], original['features'], atol=1e-5, rtol=1e-4)
            # Image-quality validity controls missing pixels; no task mask enters this API.
            valid = observation.pixel_valid.clone()
            valid[:, 1:5, 1:5] = False
            masked = replace(observation, pixel_valid=valid)
            poisoned = replace(masked, x=torch.where(valid, masked.x, float('nan')))
            clean = run(collate_optical([masked]))
            dirty = run(collate_optical([poisoned]))
            torch.testing.assert_close(clean['features'], dirty['features'], atol=0, rtol=0)
            assert not dirty['valid'][:, 1:5, 1:5].any()
            assert not dirty['features'][:, :, 1:5, 1:5].any()
            unknown = replace(observation, band_metadata=torch.zeros_like(observation.band_metadata),
                              known_mask=torch.zeros_like(observation.known_mask))
            run(collate_optical([unknown]))
            odd = replace(observation, x=observation.x[:, :-1, :-3], pixel_valid=observation.pixel_valid[:, :-1, :-3])
            run(collate_optical([odd]))
            individual.append(original)
            checks.append(dict(input=name, bands=list(observation.band_ids), shape=list(observation.x.shape),
                               permutation_max_abs=float((reverse['features']-original['features']).abs().max()),
                               padding_max_abs=float((padding['features']-original['features']).abs().max()),
                               invalid_pixels_zero=True, unknown_metadata_finite=True, odd_grid_preserved=True))
        mixed = run(collate_optical([obs for _, obs in observations]))
        for i, expected in enumerate(individual):
            torch.testing.assert_close(mixed['features'][i:i+1], expected['features'], atol=1e-5, rtol=1e-4)
        return dict(passed=True, use_sha256=reference, checks=checks, mixed_sensor_and_band_batch=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    folder = OUT/'pretraining'/args.variant
    target = folder/'interface_acceptance.json'
    if target.exists():
        raise FileExistsError('Preserve existing interface acceptance: '+str(target))
    acceptance = read(folder/'acceptance.json')
    if acceptance['passed'] is not True or sha(folder/'best.pt') != acceptance['checkpoints']['best']['file_sha256']:
        raise ValueError('Accepted best checkpoint required')
    splits, stats, _, _ = inputs('multisensor')
    observations = []
    sources = []
    for sensor in ('s2', 'landsat'):
        row = splits['TRAIN'][sensor][0]
        path = EXTERNAL/row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Accepted input cache changed')
        with np.load(path, allow_pickle=False) as arrays:
            x, valid, metadata, known = (arrays[k].copy() for k in ('x', 'pixel_valid', 'metadata', 'known'))
        mu, sd = (np.asarray(stats[sensor][k], dtype='float32') for k in ('mean', 'std'))
        x = (x-mu[:, None, None])/sd[:, None, None]
        obs = OpticalObservation(torch.from_numpy(x), tuple(row['band_ids']), torch.from_numpy(metadata),
            torch.from_numpy(known), torch.ones(len(x), dtype=torch.bool), torch.from_numpy(valid),
            dict(crs=row['crs'], transform=row['transform']))
        if sensor == 's2':
            for name, indices in [('rgb', [0,1,2]), ('rgb_nir', [0,1,2,6]),
                                  ('six', [0,1,2,6,8,9]), ('five', [0,1,2,7,8]), ('full', list(range(10)))]:
                observations.append(('s2_'+name, obs.subset([obs.band_ids[i] for i in indices])))
        else:
            observations.append(('landsat_six', obs))
        sources.append(dict(file=row['file'], sha256=row['sha256'], role=row['role'], sensor=sensor))
    saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    model = initialize(args.variant, args.device)
    model.use.load_state_dict(saved['use'], strict=True)
    frozen = FrozenUSE(model.use)
    if state_sha256(model.use) != acceptance['checkpoints']['best']['use_sha256']:
        raise ValueError('USE state differs from accepted checkpoint')
    proof = check_interface(frozen, observations, args.device)
    proof.update(variant=args.variant, checkpoint_sha256=sha(folder/'best.pt'), sources=sources,
                 auditor_sha256=sha(Path(__file__)), updates=0,
                 limitations=['Engineering interface acceptance only; no transfer or sensor generalization claim',
                              'Five-band combination was not held out from SSL masking'])
    target.write_text(json.dumps(proof, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(variant=args.variant, passed=True, inputs=len(observations)), ensure_ascii=False))


if __name__ == '__main__':
    main()
