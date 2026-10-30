"""Versioned USE input for local admissions and completed sensor counterparts."""
import json
from pathlib import Path

import numpy as np

from .use_admission import sha256


class IntegratedUSEData:
    def __init__(self, release, role='TRAIN', sensor=None, partition='all'):
        if role not in ('TRAIN', 'PRETRAIN_VAL'):
            raise ValueError('GUARD and pending sources cannot be loaded')
        self.directory = Path(release)
        contract = json.loads((self.directory/'release.json').read_text(encoding='utf8'))
        if not contract.get('ok') or sha256(self.directory/'manifest.jsonl') != contract['manifest_sha256']:
            raise ValueError('Unaccepted or modified release')
        rows = [json.loads(line) for line in (self.directory/'manifest.jsonl').read_text(encoding='utf8').splitlines()]
        self.rows = [r for r in rows if r['role']==role and (sensor is None or r['sensor']==sensor)]
        if partition not in ('all','legacy','new'):
            raise ValueError('Unknown provenance partition')
        if partition!='all':
            self.rows=[r for r in self.rows if r['id'].startswith('old_')==(partition=='legacy')]
        if not self.rows:
            raise ValueError('No admitted observations for requested role/sensor')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        from .optical_observation import OpticalObservation
        row = self.rows[index]
        path = Path(row['cache_uri'])
        if sha256(path) != row['sha256']:
            raise ValueError('Admitted cache changed')
        with np.load(path, allow_pickle=False) as arrays:
            x, valid, metadata, known = [arrays[k].copy() for k in ('x','pixel_valid','metadata','known')]
        spatial = {k:row.get(k) for k in ('sensor','crs','transform','bbox','date','role','origin','site_id','units')}
        if row['sensor'] in ('s2','landsat'):
            return OpticalObservation(torch.from_numpy(np.where(valid,x,0)),tuple(row['band_ids']),
                torch.from_numpy(metadata),torch.from_numpy(known),torch.ones(len(x),dtype=torch.bool),
                torch.from_numpy(valid),spatial).validate()
        # SAR and photographic DN retain their physical domain, outside optical wavelength API.
        return dict(x=torch.from_numpy(x), pixel_valid=torch.from_numpy(valid),
                    band_ids=tuple(row['band_ids']), spatial_metadata=spatial)


class IntegratedUSEPairs:
    """Pairs require a shared grid and >=80% joint valid pixels, with dates explicit."""
    def __init__(self, release, role='TRAIN'):
        self.data=IntegratedUSEData(release,role)
        path=Path(release)
        contract=json.loads((path/'release.json').read_text(encoding='utf8'))
        if sha256(path/'pairs.jsonl')!=contract['pairs_sha256']:
            raise ValueError('Pair membership changed')
        self.pairs=[json.loads(line) for line in (path/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
        self.pairs=[p for p in self.pairs if p['role']==role and p['qualified']]
        self.index={r['id']:i for i,r in enumerate(self.data.rows)}

    def __len__(self):return len(self.pairs)

    def __getitem__(self,index):
        pair=self.pairs[index]
        return dict(s2=self.data[self.index[pair['s2_id']]],
                    landsat=self.data[self.index[pair['landsat_id']]],pair_metadata=pair)
