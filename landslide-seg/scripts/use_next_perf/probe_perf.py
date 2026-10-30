"""Synchronized zero-optimizer profiling; never mutates production files or locks."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from use_next_p2 import run as engine
from landslide_seg.models.use_next_p2 import encoder,readout
from landslide_seg.models.use_next_p2.tasks import ScaleTask
from landslide_seg.models.use_next_v2.readout import state_digest
from use_next_perf.geometry_cache import GeometryMemo


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    lock=out/'probe.lock'
    with lock.open('x') as f:json.dump(dict(pid=os.getpid(),scope='zero optimizer performance probe'),f)
    torch.set_num_threads(4);torch.manual_seed(1234);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.mem_get_info(args.device)[0]>4*1024**3
    records=defaultdict(list);phase='initialization'
    def timed(name,fn):
        def call(*a,**kw):
            torch.cuda.synchronize(args.device);start=time.perf_counter()
            result=fn(*a,**kw)
            torch.cuda.synchronize(args.device);records[phase+'/'+name].append(time.perf_counter()-start)
            return result
        return call
    encoder.area_grid=timed('geometry_miss',encoder.area_grid)
    readout.to_label_grid=timed('label_reprojection',readout.to_label_grid)
    def instrument(model):
        for name in ('local','dofa','anysat'):
            module=getattr(model,name);module.forward=timed(name,module.forward)
    try:
        source=ROOT/'outputs/use_next_v2/runs/p2_pixel/last.pt'
        checkpoint=timed('checkpoint_read',lambda:torch.load(source,map_location='cpu',weights_only=False))()
        data=engine.IntegratedUSEData(ROOT/'outputs/use_next_v2/data/data_release_v1');index={r['id']:i for i,r in enumerate(data.rows)}
        schedule=[json.loads(l) for l in (ROOT/'outputs/use_next_v2/data/schedule_seed1234.jsonl').read_text().splitlines()]
        start=min(checkpoint['step'],5999)*8;events=schedule[start:start+8]
        assets=[timed('asset_read',engine.load_asset)(data,index[e['asset_id']]) for e in events]
        calibration=json.loads((ROOT/'outputs/use_next_v2/p2_engineering/calibration.json').read_text())
        model=ScaleTask(ROOT,'pixel').to(args.device).train();engine.restore_model(model,checkpoint['model']);instrument(model.use)
        before=state_digest(model);losses={}
        original_forward=model.use.forward
        for phase in ('pretrain_original_cold','pretrain_original_warm','pretrain_local_only_warm','pretrain_memo_fill','pretrain_memo_warm'):
            if phase=='pretrain_memo_fill':
                model.use.forward=original_forward
                encoder.area_grid=GeometryMemo(encoder.area_grid);encoder._geometry.clear()
            if phase=='pretrain_local_only_warm':
                # Exact original masking before the local encoder; only unused native outputs are omitted.
                def local_only(asset,grid,hidden=None,teacher=False):
                    if hidden is not None:asset=encoder.mask_asset(asset,hidden,grid)
                    return model.use.local(asset,grid=grid)
                model.use.forward=local_only
            result=[]
            with torch.no_grad():
                for asset,event in zip(assets,events):
                    loss,detail=timed('loss_total',model.loss)(engine.move(asset,args.device),event['geometry_mask_seed'],event['band_mask_seed'],calibration)
                    result.append(None if loss is None else loss.detach().cpu())
            losses[phase]=result
        for a,b in zip(losses['pretrain_original_warm'],losses['pretrain_local_only_warm']):
            assert (a is None and b is None) or torch.equal(a,b)
        for name in ('pretrain_memo_fill','pretrain_memo_warm'):
            for a,b in zip(losses['pretrain_original_warm'],losses[name]):
                assert (a is None and b is None) or torch.equal(a,b)
        assert state_digest(model)==before
        phase='serialization'
        timed('full_checkpoint_save',engine.atomic_save)(checkpoint,out/'disposable_checkpoint.pt')
        timed('full_curve_json',lambda:'\n'.join(json.dumps(r) for r in checkpoint['history']))()
        del model;torch.cuda.empty_cache()
        phase='head_initialization'
        local_source=torch.load(ROOT/'outputs/use_next_v2/runs/p2_physical/last.pt',map_location='cpu',weights_only=False)
        local={k[len('use.local.'):]:v for k,v in local_source['model'].items() if k.startswith('use.local.')}
        stats=engine.load_stats(ROOT/'manifests/v4r1/stats_core25.json')
        head=readout.ScaleReadout(ROOT,'physical',stats,local).to(args.device).train();instrument(head.use)
        rows=engine.read_manifest(ROOT/'manifests/v4r1/pool_100.jsonl');train=engine.USESegDataset([r for r in rows if r['split']=='train'],stats,augment=False)
        batch=timed('supervised_read',engine.collate)([train[0],train[1]])
        frozen=state_digest(head.use)
        for phase in ('head_cold','head_warm'):
            with torch.no_grad():timed('forward_total',head)(batch['image'].to(args.device))
            assert timed('full_USE_SHA',state_digest)(head.use)==frozen
        result=dict(status='ZERO_OPTIMIZER_PROFILE_COMPLETE',optimizer_updates=0,formal_updates=0,
            pretrain_cursor_start=start,sample_ids=[e['asset_id'] for e in events],local_only_loss_bitwise_equal=True,
            seconds={k:dict(calls=len(v),total=sum(v),mean=sum(v)/len(v)) for k,v in records.items()},
            limitations=['Nested inclusive timers overlap; do not sum categories','CUDA synchronization adds overhead',
                'Shared GPU, no backward/optimizer, two head examples: not production throughput or cost',
                'Loss equivalence only; gradient/RNG/full optimizer recovery acceptance still required before any optimization deployment'],
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    finally:lock.unlink()


if __name__=='__main__':main()
