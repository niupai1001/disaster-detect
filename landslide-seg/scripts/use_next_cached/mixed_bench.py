"""32 scheduled observations, actual joint backward, zero optimizer updates."""
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from use_next_cached import engine
from landslide_seg.models.use_next_p3.tasks import LatentTask
from landslide_seg.models.use_next_p2 import encoder
from use_next_perf.disk_geometry import DiskGeometry
from use_next_perf.geometry_cache import GeometryMemo
import torch


def main():
    out=ROOT/'outputs/use_next_v2/cached_p3/mixed_bench';out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(1234);engine.np.random.seed(1234)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:1';assert torch.cuda.mem_get_info(device)[0]>4*1024**3
    model=LatentTask(ROOT,'physical','joint').to(device).train()
    data=engine.IntegratedUSEData(ROOT/'outputs/use_next_v2/data/data_release_v1');ids={r['id']:i for i,r in enumerate(data.rows)}
    events=[json.loads(l) for l in (ROOT/'outputs/use_next_v2/data/schedule_seed1234.jsonl').read_text().splitlines()][:32]
    calibration=json.loads((ROOT/'outputs/use_next_v2/p2_engineering/calibration.json').read_text())
    original=encoder.area_grid;algorithm=hashlib.sha256((ROOT/'src/landslide_seg/models/use_next_p2/geometry.py').read_bytes()).hexdigest()
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();results={};reference=[]
    for name in ('original','disk_cold','disk_warm'):
        disk=DiskGeometry(original,out/'geometry',algorithm);memo=GeometryMemo(disk)
        encoder.area_grid=original if name=='original' else memo;encoder._geometry.clear()
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        torch.cuda.synchronize(device);began=time.perf_counter();batches=[]
        for batch_index in range(4):
            model.zero_grad(set_to_none=True);losses=[]
            for begin in range(batch_index*8,batch_index*8+8,2):
                terms=[]
                for event in events[begin:begin+2]:
                    asset=engine.move(engine.load_asset(data,ids[event['asset_id']]),device)
                    loss,_=model.loss(asset,event['geometry_mask_seed'],event['band_mask_seed'],calibration)
                    if loss is not None:terms.append(loss);losses.append(float(loss.detach()))
                if terms:(sum(terms)/8).backward()
            digest=engine.tensor_digest({n:p.grad for n,p in model.named_parameters() if p.grad is not None})
            row=dict(losses=losses,gradient_sha256=digest);batches.append(row)
            if name=='original':reference.append(row)
            else:assert row==reference[batch_index]
        torch.cuda.synchronize(device)
        results[name]=dict(total_seconds=time.perf_counter()-began,batches=4,observations=32,
            disk_hits=disk.hits,disk_misses=disk.misses,ram_cache_bytes=memo.bytes,ram_entries=len(memo.entries),
            disk_bytes=sum(p.stat().st_size for p in (out/'geometry').glob('*.pt')),loss_gradient_bitwise=True)
    result=dict(status='MIXED_STREAM_ZERO_OPTIMIZER_COMPLETE',optimizer_updates=0,formal_updates=0,results=results,
        unique_asset_ids=len({e['asset_id'] for e in events}),unique_disk_geometry_keys=len(list((out/'geometry').glob('*.pt'))),
        limitation='Shared GPU1; includes file reads, cache creation/loading, backward and gradient digest. Cold includes cache build cost. Warm replays same32 and is reuse upper bound, not full-training speedup.')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
