"""Minimal pre-G5 checks: actual mask stream, native bank, geometry and precision."""
from dataclasses import replace
import itertools
import time

import numpy as np
import torch

from .encoder_v2 import EncoderConfig,ObservationEncoderV2,OpticalBand,OpticalImage,TargetGrid
from .encoder_v2_tasks import (training_spatial_task,validation_spatial_task,training_spectral_keys,
    held_out_combinations,spectral_task,run_task,balanced_loss,input_hash)
from .encoder_v2_runtime import BASE,G3,load_members
from .check_encoder_v2 import state_hash
from .expansion_data import OUT,read,save

DEST=BASE/'pre_G5'


def main():
    DEST.mkdir(exist_ok=True);start=time.perf_counter();torch.manual_seed(0);torch.set_num_threads(4)
    config=EncoderConfig(local_width=16,query_chunk=2048)
    model=ObservationEncoderV2(config).eval();before=state_hash(model)
    data=load_members(read(G3/'members.json'));meta,image=data[0]
    a=training_spatial_task(image,meta['id'],1);b=training_spatial_task(image,meta['id'],2)
    assert input_hash(a)!=input_hash(b) and input_hash(a)==input_hash(training_spatial_task(image,meta['id'],1))
    assert input_hash(validation_spatial_task(image,meta['id']))==input_hash(validation_spatial_task(image,meta['id']))
    for update in range(1,201):
        hidden=training_spectral_keys(image,meta['id'],update)
        assert not any(set(keys)<=set(hidden) for _,keys in held_out_combinations(image))
    with torch.no_grad():
        representation=model.encode(a.visible);bank=representation.optional_local_features[0][0]
        generator=torch.Generator().manual_seed(0);coordinates=torch.rand(16,2,generator=generator)*1000
        found=model.decoder.nearest_native(bank,coordinates,representation.grids[0]['bounds'])
        distances=torch.cdist(coordinates,bank.relative_xy_m,compute_mode='donot_use_mm_for_euclid_dist')
        expected=distances.topk(4,largest=False).values
        actual=(bank.relative_xy_m[found]-coordinates[:,None]).square().sum(-1).sqrt()
        assert torch.allclose(actual,expected,atol=1e-4,rtol=1e-5)
    # Same nullspace perturbation as the old information-loss witness.
    arrays=np.load(DEST/'information_check/counterexample.npz')
    small_source=image.bands[0]
    first=replace(small_source,values=torch.from_numpy(arrays['first']),valid=torch.ones(120,120,dtype=torch.bool),
        visible=None,transform=(10.,0.,500000.,0.,-10.,4000000.),band_key='visible')
    second=replace(first,values=torch.from_numpy(arrays['second']))
    with torch.no_grad():
        one=model.encode(OpticalImage([first]));two=model.encode(OpticalImage([second]))
        feature_delta=float((one.optional_local_features[0][0].features-two.optional_local_features[0][0].features).abs().max())
        assert feature_delta>1e-4
    tiny=replace(first,values=first.values[:12,:12].clone(),valid=first.valid[:12,:12])
    target=TargetGrid((12,12),tiny.transform,tiny.crs,tiny.srf,'target')
    ids=torch.tensor([0,12,55,71,143])
    with torch.no_grad():
        rep=model.encode(OpticalImage([tiny]));full,_=model.decoder(rep,target);sample,_=model.decoder(rep,target,pixel_indices=ids)
        assert torch.allclose(sample,full.flatten()[ids],atol=1e-5,rtol=1e-5)
        reversed_rep=model.encode(OpticalImage(list(reversed(a.visible.bands))))
        y,_=model.decoder(representation,TargetGrid(tuple(image.bands[0].values.shape),image.bands[0].transform,image.bands[0].crs,image.bands[0].srf,'query'),pixel_indices=torch.arange(64))
        yr,_=model.decoder(reversed_rep,TargetGrid(tuple(image.bands[0].values.shape),image.bands[0].transform,image.bands[0].crs,image.bands[0].srf,'query'),pixel_indices=torch.arange(64))
        assert torch.allclose(y,yr,atol=1e-5,rtol=1e-5)
        altered=OpticalImage([replace(b,values=torch.where(b.visible,b.values,b.values+10000),valid=torch.where(b.visible,b.valid,~b.valid)) for b in a.visible.bands])
        changed=model.encode(altered)
        assert torch.equal(representation.features,changed.features)
        assert all(torch.equal(x.features,z.features) for x,z in zip(representation.optional_local_features[0],changed.optional_local_features[0]))
        task=spectral_task(image,['B11','B12'])
        altered_image=OpticalImage([replace(b,values=b.values+10000,valid=~b.valid) if b.band_key in ['B11','B12'] else b for b in image.bands])
        assert input_hash(task)==input_hash(spectral_task(altered_image,['B11','B12']))
    values=torch.tensor([1.,4.,9.,16.]);means=[values[list(pair)].mean() for pair in itertools.combinations(range(4),2)]
    assert torch.stack(means).mean()==values.mean()
    rows,exposure,_=run_task(model,a,meta,read(OUT/'statistics.json')['bands'],target_sample_pixels=128,logical_update=1)
    loss=balanced_loss(rows);loss.backward()
    norm=float(torch.stack([p.grad.square().sum() for n,p in model.named_parameters() if 'local_' in n and p.grad is not None]).sum().sqrt())
    assert norm>0 and np.isfinite(norm) and before==state_hash(model)
    cpu=dict(status='PASS',new_local_feature_delta_on_old_nullspace=feature_delta,
        old_global_H_delta=read(DEST/'information_check/original.json')['encoder_H_max_change'],
        exact_native_neighbors_match_brute_force=True,masked_QA_values_zero_leakage=True,
        multi_band_target_excluded=True,held_out_combinations_never_sampled_in_training=True,
        train_resampled_validation_fixed_resume_exact=True,pixel_subset_matches_full_decoder=True,
        uniform_target_estimator_hand_check=True,local_gradient_norm=norm,optimizer_updates=0,
        native_local_features_accounted=exposure['local_feature_vectors'],seconds=time.perf_counter()-start)
    save(DEST/'engineering_checks.json',cpu)
    # One real mixed-precision comparison, plus timing. No optimizer object.
    device='cuda:1';torch.cuda.set_device(device);model=model.to(device)
    gpu_data=load_members([read(G3/'members.json')[0]],device);meta,image=gpu_data[0]
    task=spectral_task(image,['B11','B12']);stats=read(OUT/'statistics.json')['bands'];comparisons={}
    for precision in ['fp32','bf16']:
        model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            rows,exposure,_=run_task(model,task,meta,stats,target_sample_pixels=512,logical_update=1)
            loss=balanced_loss(rows)
        loss.backward();torch.cuda.synchronize()
        vector=torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()]).float()
        assert torch.isfinite(vector).all() and torch.isfinite(loss)
        comparisons[precision]=dict(loss=float(loss.detach()),gradient=vector.detach().clone(),
            seconds=time.perf_counter()-tick,peak_bytes=torch.cuda.max_memory_allocated())
    relative_loss=abs(comparisons['bf16']['loss']/comparisons['fp32']['loss']-1)
    relative_gradient=float((comparisons['bf16']['gradient']-comparisons['fp32']['gradient']).norm()/comparisons['fp32']['gradient'].norm())
    for row in comparisons.values():row.pop('gradient')
    accepted=relative_loss<=.01 and relative_gradient<=.05
    save(DEST/'precision_and_timing.json',dict(status='PASS',chosen_precision='bf16' if accepted else 'fp32',
        bf16_accepted=accepted,relative_loss=relative_loss,relative_gradient=relative_gradient,measurements=comparisons,
        scope='One actual SWIR-pair prediction,512 uniform valid targets per band; not full effective-batch throughput',optimizer_updates=0))
    assert before==state_hash(model)
    print('PRE_G5_PASS',feature_delta,relative_loss,relative_gradient,comparisons,flush=True)


if __name__=='__main__':main()
