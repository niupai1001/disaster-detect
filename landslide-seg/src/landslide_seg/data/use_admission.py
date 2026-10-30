"""Read only released USE observations; pending sources never enter the loader."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class AdmittedUSEData:
    """Physical reflectance by default; normalization is explicit and sensor-specific."""
    def __init__(self, release, role='TRAIN', sensor=None, normalize=False):
        if role not in ('TRAIN', 'PRETRAIN_VAL'):
            raise ValueError('Only TRAIN or PRETRAIN_VAL may be loaded')
        self.release = Path(release)
        contract = json.loads((self.release / 'release.json').read_text(encoding='utf8'))
        if contract['ok'] is not True:
            raise ValueError('Release is not accepted')
        self.root = Path(contract['project_root'])
        for name, expected in contract['artifacts'].items():
            if sha256(self.release / name) != expected:
                raise ValueError('Release artifact changed: ' + name)
        for name, expected in contract['source_sha256'].items():
            if sha256(self.root / name) != expected:
                raise ValueError('Frozen source changed: ' + name)
        self.cache_root = self.root / 'outputs/use/external'
        rows = [json.loads(line) for line in (self.release / 'admitted.jsonl').read_text(encoding='utf8').splitlines()]
        self.rows = [r for r in rows if r['role'] == role and (sensor is None or r['sensor'] == sensor)]
        if not self.rows:
            raise ValueError('Empty requested split')
        self.stats = json.loads((self.release / 'normalization.json').read_text(encoding='utf8'))
        self.normalize = normalize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        from .optical_observation import OpticalObservation
        row = self.rows[index]
        path = (self.cache_root / row['file']).resolve()
        if not path.is_relative_to(self.cache_root.resolve()) or sha256(path) != row['sha256']:
            raise ValueError('Invalid or modified cache')
        with np.load(path, allow_pickle=False) as arrays:
            x, valid, metadata, known = [arrays[k].copy() for k in ('x', 'pixel_valid', 'metadata', 'known')]
        units = 'surface_reflectance'
        if self.normalize:
            stats = self.stats[row['sensor']]
            mean, std = [np.asarray(stats[k], dtype='float32')[:, None, None] for k in ('mean', 'std')]
            if len(mean) != len(x) or np.any(std <= 0):
                raise ValueError('Sensor normalization mismatch')
            x = (x - mean) / std
            units = 'frozen_sensor_zscore'
        # Invalid reflectance is never interpreted as a target or learned zero-valued feature.
        x = np.where(valid, x, 0).astype('float32')
        observation = OpticalObservation(torch.from_numpy(x), tuple(row['band_ids']),
            torch.from_numpy(metadata), torch.from_numpy(known), torch.ones(len(x), dtype=torch.bool),
            torch.from_numpy(valid), dict(sensor=row['sensor'], crs=row['crs'], transform=row['transform'],
                bbox=row['bbox'], date=row.get('datetime', row.get('date')), units=units,
                file=row['file'], role=row['role'], source_sha256=row['sha256']))
        return observation.validate()
