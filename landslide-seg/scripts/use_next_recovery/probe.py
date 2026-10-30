"""Zero-update, fixed failed-batch geometry equivalence probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from landslide_seg.data.use_integrated import IntegratedUSEData
from landslide_seg.models.use_next_v2 import load_asset
from landslide_seg.models.use_next_p2.geometry import area_grid as original
from use_next_recovery.geometry import area_grid as candidate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--repeat',type=int,default=3)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    data=IntegratedUSEData(ROOT/'outputs/use_next_v2/data/data_release_v1')
    ids={r['id']:i for i,r in enumerate(data.rows)}
    schedule=[json.loads(l) for l in (ROOT/'outputs/use_next_v2/data/schedule_seed1234.jsonl').read_text().splitlines()]
    rows=[]
    for cursor in range(1678*8,1679*8):
        event=schedule[cursor];asset=load_asset(data,ids[event['asset_id']])
        print(json.dumps(dict(cursor=cursor,id=event['asset_id'],shape=list(asset.image.shape))),flush=True)
        for repeat in range(args.repeat):
            first=original(asset);second=candidate(asset)
            assert first[1:]==second[1:]
            assert torch.equal(first[0].indices(),second[0].indices()) and torch.equal(first[0].values(),second[0].values())
            grid=dict(crs=first[3],shape=first[1],origin=(first[2][2],first[2][5]))
            shared=candidate(asset,grid)
            assert shared[1:]==first[1:] and torch.equal(shared[0].values(),first[0].values())
        rows.append(dict(cursor=cursor,id=event['asset_id'],shape=list(asset.image.shape),repeat=args.repeat,exact=True))
        (out/'progress.json').write_text(json.dumps(rows,indent=2))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='FIXED_BATCH_GEOMETRY_EXACT',formal_updates=0,optimizer_updates=0,rows=rows,
        original_sha256=sha(ROOT/'src/landslide_seg/models/use_next_p2/geometry.py'),
        candidate_sha256=sha(ROOT/'scripts/use_next_recovery/geometry.py'),
        limitation='Failure to reproduce a native crash does not establish its cause or prove long-run stability.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
