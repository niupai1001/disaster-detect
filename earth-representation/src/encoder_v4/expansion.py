"""Read the scoped data handoff. The formal runner continues to use core release v1."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .data import Attribute, Band, Observation, OpticalDataset, Physics, asset_path, observation_to
from .geometry import Grid
from .release import Isolation, POLICY, check_splits, file_sha, load_release, read_json, rows

SCHEMA = 'pd_here_optical_expansion_v1'
SCREENED_SCHEMA = 'pd_here_optical_expansion_v2'
EVAL_SCHEMA = 'pd_here_optical_expansion_v3'
EVAL_SCREEN = 'heldout_support_screen_v3'
PAIRED_SCHEMA = 'pd_here_optical_expansion_v4'
PAIRED_EVAL = 'paired_source_eval_v4'
PAIR_DOMAINS = {'flair_spot_boa_reflectance', 'flair_s2_boa_reflectance',
                'maja_registered_boa_reflectance', 'venus_s2_regressed_reflectance',
                'nbar_harmonized_reflectance', 'boa_surface_reflectance', 'rayleigh_corrected_reflectance'}
FMOW_DN_DOMAINS = {f'fmow_{instrument}_dn' for instrument in ('worldview2','worldview3','geoeye1','quickbird2')}
EVAL_DOMAINS = {'landsatbench_optical_dn', 'worldview3_dn'} | FMOW_DN_DOMAINS
SCREENED_DOMAINS = {'landsatbench_optical_dn', 'rayleigh_corrected_reflectance',
                    'worldview3_dn', 'flair_spot_boa_reflectance', 'flair_s2_boa_reflectance'} | FMOW_DN_DOMAINS
ROLES = ('TRAIN', 'VAL_EXT', 'TEST_EXT')
DOMAINS = {'boa_surface_reflectance', 'nbar_harmonized_reflectance',
           'rayleigh_corrected_reflectance', 'venus_s2_regressed_reflectance',
           'maja_registered_boa_reflectance', 'resampled_landsat_boa_reflectance'}


@lru_cache(maxsize=1024)
def physical_from_json(serialized):
    import json
    fields = {}
    for name, attr in json.loads(serialized).items():
        fields[name] = Attribute(attr['status'], attr.get('unit', ''), attr.get('source', ''),
                                 tuple(attr.get('summary', ())), tuple(map(tuple, attr.get('samples', ()))),
                                 tuple(attr.get('summary_fields', ())))
    return Physics(**fields)


def effective_physics(item, record):
    import json
    fields = item['attributes'].copy()
    if record.get('source_response_only'):
        for name in ('srf', 'psf', 'mtf'):
            fields[name] = {'status': 'missing'}
    if record.get('source_spectral_only'):
        for name in ('center', 'bandwidth'):
            fields[name] = {'status': 'missing'}
    if record.get('native_gsd_xy_m') is None:
        fields['gsd'] = {'status': 'missing'}
    else:
        fields['gsd'] = dict(status=record.get('native_gsd_status','observed'), unit='m', source=record['native_gsd_source'],
                             summary=record['native_gsd_xy_m'])
    return physical_from_json(json.dumps(fields, sort_keys=True, separators=(',', ':')))


def split_guard(member):
    return dict(id=member['id'], native_bbox=member['native_bbox'],
                canonical_acquisition=member['acquisition_id'] or 'unknown:' + member['id'],
                acquisition_aliases=member.get('acquisition_aliases',()),
                observation_key=member['observation_id'], cache_sha256=member['content_sha256'],
                isolation_group=member['geographic_group'])


def verify_paired_domains(roles):
    def domains(group):
        return {(r['dataset'], r['radiometric_domain']) for r in group}
    train = domains(roles['TRAIN'])
    evaluation = domains(roles['VAL_EXT']) | domains(roles['TEST_EXT'])
    if train != evaluation:
        raise ValueError(f'Unpaired product domains: TRAIN only={sorted(train-evaluation)}, evaluation only={sorted(evaluation-train)}')
    return train


class ExpansionDataset:
    """Opt-in data adapter with sealed domain/role identities, independent of optimizer configuration."""
    def __init__(self, folder, root, *, staging=False):
        self.folder, self.root = Path(folder), Path(root)
        self.release = read_json(self.folder / 'release.json')
        version = self.release.get('schema')
        if version not in (SCHEMA, SCREENED_SCHEMA, EVAL_SCHEMA, PAIRED_SCHEMA) or self.release.get('training_authorized') is not False:
            raise ValueError('Unsupported scoped data release')
        required = {'extension_members.jsonl', 'physical_metadata.jsonl', 'core_physical_index.json', 'diagnostic_panel.jsonl', 'acceptance.json'}
        if version in (EVAL_SCHEMA, PAIRED_SCHEMA):
            required |= {'val_ext_panel.npz', 'test_ext_panel.npz'}
        if not staging and not required <= self.release['files'].keys():
            raise ValueError('Incomplete expansion seal')
        for name, digest in self.release['files'].items():
            path = self.folder / name
            if not path.resolve().is_relative_to(self.folder.resolve()) or file_sha(path) != digest:
                raise ValueError(f'Release file identity mismatch: {name}')
        core = asset_path(self.root, self.release['core']['path'])
        if file_sha(core / 'release.json') != self.release['core']['sha256']:
            raise ValueError('Core publication changed')
        self.core_roles, _ = load_release(core)
        self.core = OpticalDataset(self.root)
        self.metadata = {r['id']: r for r in rows(self.folder / 'physical_metadata.jsonl')}
        self.core_physics = read_json(self.folder / 'core_physical_index.json')
        self.roles = {role: [] for role in ROLES}
        for row in rows(self.folder / 'extension_members.jsonl'):
            policy = row.get('quality_policy')
            screened = policy in ('train_support_screen_v2', EVAL_SCREEN, PAIRED_EVAL)
            if policy == PAIRED_EVAL and (version != PAIRED_SCHEMA or row['split_role'] not in ROLES[1:]
                                           or row['radiometric_domain'] not in PAIR_DOMAINS
                                           or row.get('value_encoding') != 'physical'
                                           or row.get('evaluation_scope') != 'same_product_physical'):
                raise ValueError('Invalid physical product evaluation contract')
            if policy == EVAL_SCREEN and (version not in (EVAL_SCHEMA, PAIRED_SCHEMA) or row['split_role'] not in ROLES[1:]
                                          or row['radiometric_domain'] not in EVAL_DOMAINS
                                          or row.get('value_encoding') != 'stored_dn'
                                          or row.get('evaluation_scope') != 'same_product_dn'):
                raise ValueError('Invalid support-screened evaluation contract')
            if policy == 'train_support_screen_v2' and (version not in (SCREENED_SCHEMA, EVAL_SCHEMA, PAIRED_SCHEMA) or row['split_role'] != 'TRAIN'
                             or row['radiometric_domain'] not in SCREENED_DOMAINS
                             or row.get('value_encoding') not in ('physical', 'stored_dn')):
                raise ValueError('Support-screened data requires the versioned TRAIN contract')
            if (row['radiometric_domain'] not in (DOMAINS | SCREENED_DOMAINS | PAIR_DOMAINS if screened else DOMAINS) or
                    row['admission_status'] != 'extension_optical' or
                    row['qa_scope'] not in (('patch', 'pixel', 'none') if screened else ('patch', 'pixel')) or
                    not row['eligible_tasks'] or not set(row['eligible_tasks']) <= {'A', 'B1'}):
                raise ValueError(f'Unsupported member contract: {row["id"]}')
            if not staging and self.release['files'].get(row['cache']) != row['cache_sha256']:
                raise ValueError('Cache is not bound by the release')
            self.roles[row['split_role']].append(row)
        role_aliases = dict(TRAIN='TRAIN', VAL_EXT='VAL', TEST_EXT='TEST')
        from collections import Counter
        if {role: dict(Counter(r['dataset'] for r in group)) for role, group in self.roles.items()} != self.release['counts']:
            raise ValueError('Scoped role counts disagree with release')
        guards = {role_aliases[k]: [split_guard(r) for r in values] for k, values in self.roles.items()}
        check_splits(guards, POLICY['heldout_buffer_degrees'])
        for role, records in guards.items():
            guard = Isolation((r for key, group in self.core_roles.items() if key != role for r in group),
                              POLICY['heldout_buffer_degrees'])
            for row in records:
                if conflict := guard.conflict(row):
                    raise ValueError(f'Extension/core role collision: {row["id"]}: {conflict}')

        if version == PAIRED_SCHEMA:
            verify_paired_domains(self.roles)
        self.verified = {}

    def panel_path(self, role):
        if self.release['schema'] not in (EVAL_SCHEMA, PAIRED_SCHEMA) or role not in ROLES[1:]:
            raise ValueError('A frozen extension evaluation role is required')
        return self.folder / f'{role.lower()}_panel.npz'

    def load_core(self, row, device='cpu'):
        from dataclasses import replace
        obs = self.core.load(row, device)
        bands = []
        for band in obs.bands:
            key = self.core_physics.get(row['id'], {}).get(band.key)
            if key:
                phys = effective_physics(self.metadata[key], dict(
                    native_gsd_xy_m=band.grid.spacing, native_gsd_source='frozen core native affine'))
                band = replace(band, physics=phys)
            bands.append(band)
        return replace(obs, bands=tuple(bands))

    def load(self, row, device='cpu'):
        from encoder_v3.data import checked_file
        path = self.folder / row['cache']
        if not path.resolve().is_relative_to(self.folder.resolve()):
            raise ValueError(f'Cache path outside release: {row["id"]}')
        checked_file(path, row['cache_sha256'], self.verified)
        bands = []
        with np.load(path, allow_pickle=False) as saved:
            for rec in row['bands']:
                key = rec['key']
                values, valid = saved[key], saved[key + '__valid']
                grid = Grid(tuple(rec['shape']), tuple(rec['affine']), rec['crs'])
                is_physical = row.get('value_encoding', 'physical') == 'physical'
                if (rec.get('calibrated_already') is not is_physical or values.dtype != np.float32 or
                        valid.dtype != np.bool_ or values.shape != grid.shape or valid.shape != grid.shape or
                        not np.isfinite(values[valid]).all()):
                    raise ValueError(f'Cache calibration, shape or QA mismatch: {row["id"]}/{key}')
                phys = effective_physics(self.metadata[rec['physical_id']], rec)
                bands.append(Band(key, torch.from_numpy(values), torch.from_numpy(valid), grid, phys))
        if not bands or len({b.key for b in bands}) != len(bands):
            raise ValueError(f'Expected unique bands: {row["id"]}')
        return observation_to(Observation(row['id'], row['instrument'], tuple(bands)), device) 

