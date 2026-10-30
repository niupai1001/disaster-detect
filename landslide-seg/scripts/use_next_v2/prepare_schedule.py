"""Freeze a shared TRAIN-only observation schedule; never selects by task outcomes."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from landslide_seg.models.use_next_v2.physical import PhysicalLocal


def main():
    release=ROOT/'outputs/use_next_v2/data/data_release_v1'
    contract=json.loads((release/'release.json').read_text())
    groups=defaultdict(lambda:defaultdict(list))
    for line in (release/'manifest.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['role']!='TRAIN':continue
        site=r.get('site_id') or r.get('group') or json.dumps(r.get('cell')) if r.get('cell') else r.get('site_id') or r.get('group') or r['v2_geographic_group']
        groups[r['sensor']][site].append(r['id'])
    sensors=sorted(groups);prob=[contract['modality_probabilities'][s] for s in sensors]
    rng=np.random.default_rng(1234)
    out=ROOT/'outputs/use_next_v2/data/schedule_seed1234.jsonl'
    with out.open('x') as f:
        for exposure in range(10000*8):
            sensor=str(rng.choice(sensors,p=prob));sites=sorted(groups[sensor]);site=sites[int(rng.integers(len(sites)))];ids=sorted(groups[sensor][site])
            f.write(json.dumps(dict(exposure=exposure,sensor=sensor,site=site,asset_id=ids[int(rng.integers(len(ids)))],
                                   geometry_mask_seed=int(rng.integers(2**31)),band_mask_seed=int(rng.integers(2**31))))+'\n')
    torch.manual_seed(1234);model=PhysicalLocal()
    torch.save(dict(local=model.state_dict(),seed=1234,trained_updates=0),ROOT/'outputs/use_next_v2/engineering/u0.pt')
    print(out)


if __name__=='__main__':main()
