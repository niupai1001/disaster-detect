"""Fixed-endpoint representation gate, without label access or encoder selection."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from landslide_seg.data.use_integrated import IntegratedUSEData
from landslide_seg.models.use_next_v2 import load_asset
from landslide_seg.models.use_next_p2.encoder import ScaleEncoder,grid_for
from use_next_p2.run import move
from use_next_p2.protocol import sha
from use_next_v2.device import claim
from contextlib import nullcontext


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',default='cpu');ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if not args.engineering and (saved['step']!=10000 or saved['identity']['engineering']):raise ValueError('P3 fixed formal u10000 required')
    context=claim(args.device) if args.device.startswith('cuda') else nullcontext('cpu')
    torch.set_num_threads(4)
    with context as uuid:
        encoder=ScaleEncoder('physical').to(args.device).eval().requires_grad_(False)
        local={k[len('use.local.'):]:v for k,v in saved['model'].items() if k.startswith('use.local.')}
        encoder.load_state_dict(local,strict=True)
        teacher=copy.deepcopy(encoder)
        teacher.load_state_dict({k[len('use.teacher_local.'):]:v for k,v in saved['model'].items()
                                 if k.startswith('use.teacher_local.')},strict=True)
        records=[];embeddings=[];teacher_embeddings=[]
        for sensor in ('s2','landsat','s1'):
            dataset=IntegratedUSEData(ROOT/'outputs/use_next_v2/data/data_release_v1',role='PRETRAIN_VAL',sensor=sensor)
            for partition in ('legacy599','additional'):
                indices=[i for i,r in enumerate(dataset.rows) if r['validation_partition']==partition][:4]
                for index in indices:
                    asset=move(load_asset(dataset,index),args.device);grid=grid_for(asset)
                    with torch.no_grad():
                        output=encoder(asset,grid=grid)
                        target=teacher(asset,grid=grid)
                        zero=copy.copy(asset);zero.image=torch.zeros_like(asset.image)
                        other=encoder(zero,grid=grid)
                    feature=output.metric_patch.tensor[:,output.metric_patch.valid_mask].T
                    variance=float((feature-feature.mean(0)).square().mean())
                    teacher_feature=target.metric_patch.tensor[:,target.metric_patch.valid_mask].T
                    teacher_variance=float((teacher_feature-teacher_feature.mean(0)).square().mean())
                    delta=float((output.global_embedding-other.global_embedding).norm())
                    assert torch.isfinite(feature).all() and variance>1e-8
                    assert torch.isfinite(teacher_feature).all() and teacher_variance>1e-8
                    embeddings.append(output.global_embedding.cpu())
                    teacher_embeddings.append(target.global_embedding.cpu())
                    records.append(dict(id=asset.provenance['id'],sensor=sensor,partition=partition,spatial_variance=variance,
                                        teacher_spatial_variance=teacher_variance,content_delta=delta))
        values=torch.stack(embeddings);centered=values-values.mean(0);covariance=centered.T@centered/(len(values)-1)
        covariance_trace=float(covariance.trace());maximum_delta=max(r['content_delta'] for r in records)
        teacher_values=torch.stack(teacher_embeddings);teacher_centered=teacher_values-teacher_values.mean(0)
        teacher_covariance=teacher_centered.T@teacher_centered/(len(teacher_values)-1)
        teacher_covariance_trace=float(teacher_covariance.trace())
        passed=(covariance_trace>1e-8 and teacher_covariance_trace>1e-8 and maximum_delta>1e-6
                and torch.isfinite(covariance).all().item() and torch.isfinite(teacher_covariance).all().item())
        result=dict(status='P3_REPRESENTATION_GATE_PASS' if passed else 'P3_REPRESENTATION_GATE_FAILED',
                    formal_fixed_endpoint=not args.engineering,checkpoint_sha256=sha(args.checkpoint),step=saved['step'],uuid=uuid,
                    covariance_trace=covariance_trace,teacher_covariance_trace=teacher_covariance_trace,
                    maximum_content_delta=maximum_delta,records=records,
                    limitation='Engineering nondegeneracy/content dependence only; does not prove semantic transfer. No encoder selection or budget extension.')
        Path(args.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
