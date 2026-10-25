"""Measure actual frozen-encoder latency and memory on existing sensor caches."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from landslide_seg.models.frozen_use import FrozenUSE
from landslide_seg.data.optical_observation import collate_optical
from pretrain_use_independent import OUT, initialize, inputs
from audit_use_independent_data import read, sha
from evaluate_use_independent_ssl import observation


def attention_counts(channels, height, width, config):
    positions = math.ceil(height/config.get('relation_grid',2))*math.ceil(width/config.get('relation_grid',2))
    heads, dimension, queries = config.get('heads',4), config.get('width',32), config.get('queries',4)
    self_pairs = positions*channels*channels if config.get('attention',True) else 0
    pool_pairs = positions*queries*channels
    return dict(batch=1, relation_positions=positions, band_tokens=positions*channels,
        learned_query_tokens=positions*queries, attention_scores_all_heads=heads*(self_pairs+pool_pairs),
        qk_and_av_multiply_accumulates=2*dimension*(self_pairs+pool_pairs),
        cost_scope='QK and attention-value matrix products only; excludes projections, convolutions, normalization, interpolation and hashing; not total FLOPs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',required=True)
    parser.add_argument('--device',default='cuda:1')
    args = parser.parse_args()
    folder = OUT/'pretraining'/args.variant
    target = folder/'efficiency.json'
    if target.exists(): raise FileExistsError('Preserve existing benchmark')
    accepted, config = read(folder/'acceptance.json'), read(folder/'resolved_config.json')
    if not accepted['passed'] or sha(folder/'best.pt') != accepted['checkpoints']['best']['file_sha256']:
        raise ValueError('Accepted selected checkpoint required')
    splits, stats, _, _ = inputs('multisensor')
    model = initialize(args.variant,args.device)
    saved = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model.use.load_state_dict(saved['use'],strict=True)
    frozen = FrozenUSE(model.use)
    del model,saved
    digest = frozen.verify()
    if digest != accepted['checkpoints']['best']['use_sha256']: raise ValueError('Selected encoder differs')
    device = torch.device(args.device)
    if device.type != 'cuda': raise ValueError('This benchmark requires the actual CUDA device')
    results = {}
    sources = {}
    with torch.no_grad():
        for sensor in ('s2','landsat'):
            row = splits['PRETRAIN_VAL'][sensor][0]
            obs = observation(row,stats[sensor])
            sources[sensor] = dict(file=row['file'],sha256=row['sha256'],crs=row['crs'],transform=row['transform'])
            sets = (('rgb', [0,1,2]),('rgb_nir',[0,1,2,6]),('six',[0,1,2,6,8,9]),('full',list(range(10)))) if sensor == 's2' else (('six',list(range(6))),)
            for name, indices in sets:
                batch = collate_optical([obs.subset([obs.band_ids[i] for i in indices])]).to(device)
                for _ in range(5): frozen(batch)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                initial_memory = torch.cuda.memory_allocated(device)
                times = []
                for _ in range(20):
                    started = time.perf_counter()
                    frozen(batch)
                    torch.cuda.synchronize(device)
                    times.append(time.perf_counter()-started)
                mean = statistics.mean(times)
                results[sensor+'_'+name] = dict(channels=len(indices),shape=list(batch['x'].shape),
                    median_ms=statistics.median(times)*1000,mean_ms=mean*1000,tiles_per_second=1/mean,
                    baseline_allocated_mib=initial_memory/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                    incremental_peak_allocated_mib=(torch.cuda.max_memory_allocated(device)-initial_memory)/2**20,
                    attention=attention_counts(len(indices),obs.x.shape[-2],obs.x.shape[-1],config['use_config']))
                del batch
    if frozen.verify() != digest: raise ValueError('Encoder changed during benchmark')
    proof = dict(passed=True,variant=args.variant,use_sha256=digest,checkpoint_sha256=sha(folder/'best.pt'),
        device=str(device),gpu=torch.cuda.get_device_name(device),torch_version=torch.__version__,precision='FP32',
        warmup_iterations=5,measured_iterations=20,training_updates=0,results=results,sources=sources,
        auditor_sha256=sha(Path(__file__)),
        limitations=['Wall time includes production frozen-state hash checks and Python overhead',
                     'Batch1 only, one GPU and one measurement session; timings are descriptive',
                     'Each cache uses its declared common grid; native multi-resolution token processing is not implemented'])
    target.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(args.variant,'efficiency accepted',flush=True)


if __name__ == '__main__': main()
