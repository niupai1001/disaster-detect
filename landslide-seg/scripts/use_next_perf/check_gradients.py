"""Zero-optimizer cache equivalence through the real accumulated backward path."""
import json
import os
from pathlib import Path
import sys
import time
import argparse
import hashlib
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from use_next_p2 import run as engine
from landslide_seg.models.use_next_p2 import encoder
from landslide_seg.models.use_next_p2.tasks import ScaleTask
from landslide_seg.models.use_next_v2.readout import state_digest
from use_next_perf.geometry_cache import GeometryMemo
import torch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--disk',action='store_true');args=ap.parse_args()
    out=ROOT/('outputs/use_next_v2/performance/disk_gradient_check' if args.disk else 'outputs/use_next_v2/performance/gradient_check');out.mkdir(parents=True,exist_ok=False)
    lock=out/'probe.lock';lock.write_text(str(os.getpid()))
    try:
        device='cuda:0';torch.set_num_threads(4);torch.manual_seed(1234)
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert torch.cuda.mem_get_info(device)[0]>4*1024**3
        data=engine.IntegratedUSEData(ROOT/'outputs/use_next_v2/data/data_release_v1');ids={r['id']:i for i,r in enumerate(data.rows)}
        schedule=[json.loads(l) for l in (ROOT/'outputs/use_next_v2/data/schedule_seed1234.jsonl').read_text().splitlines()]
        events=schedule[33104:33112]
        assets=[engine.move(engine.load_asset(data,ids[e['asset_id']]),device) for e in events]
        calibration=json.loads((ROOT/'outputs/use_next_v2/p2_engineering/calibration.json').read_text())
        original=encoder.area_grid;results={}
        for mode in ('pixel','physical'):
            model=ScaleTask(ROOT,mode).to(device).train();before=state_digest(model)
            memo=GeometryMemo(original)
            if args.disk:
                from use_next_perf.disk_geometry import DiskGeometry
                memo=DiskGeometry(original,out/'geometry',hashlib.sha256((ROOT/'src/landslide_seg/models/use_next_p2/geometry.py').read_bytes()).hexdigest())
            encoder.area_grid=memo;encoder._geometry.clear()
            with torch.no_grad():
                for a,e in zip(assets,events):model.loss(a,e['geometry_mask_seed'],e['band_mask_seed'],calibration)
            rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
            passes={}
            for label,compute in (('original',original),('cached',memo)):
                encoder.area_grid=compute;encoder._geometry.clear()
                torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng);model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device);start=time.perf_counter();losses=[]
                for begin in range(0,8,2):
                    terms=[]
                    for a,e in zip(assets[begin:begin+2],events[begin:begin+2]):
                        loss,_=model.loss(a,e['geometry_mask_seed'],e['band_mask_seed'],calibration)
                        if loss is not None:terms.append(loss);losses.append(float(loss.detach()))
                    if terms:(sum(terms)/8).backward()
                torch.cuda.synchronize(device)
                passes[label]=dict(seconds=time.perf_counter()-start,losses=losses,
                    gradients={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()},
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
            a,b=passes['original'],passes['cached'];assert a['losses']==b['losses']
            for name,value in a['gradients'].items():
                other=b['gradients'][name]
                assert (value is None and other is None) or torch.equal(value,other),name
            assert torch.equal(a['cpu_rng'],b['cpu_rng'])
            assert all(torch.equal(x,y) for x,y in zip(a['cuda_rng'],b['cuda_rng']))
            assert state_digest(model)==before
            results[mode]=dict(loss_bitwise=True,all_gradients_bitwise=True,rng_bitwise=True,model_unchanged=True,
                original_seconds=a['seconds'],cached_seconds=b['seconds'],cache_bytes=getattr(memo,'bytes',None),
                disk_hits=getattr(memo,'hits',None),disk_misses=getattr(memo,'misses',None))
            del model;torch.cuda.empty_cache()
        encoder.area_grid=original
        result=dict(status='CACHE_LOSS_GRADIENT_RNG_EXACT',optimizer_updates=0,formal_updates=0,results=results,
                    limitation='Prewarmed repeated batch; not production hit-rate or end-to-end throughput. Full optimizer/restart acceptance still required.')
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:lock.unlink()


if __name__=='__main__':main()
