"""Bounded G2 engineering acceptance. No optimizer is constructed or stepped."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .encoder_v2 import (EncoderConfig,ObservationEncoderV2,OpticalBand,OpticalImage,
    image_from_g1,target_from_band,token_grid,pool_native,srf_descriptor)
from .expanded_dataset import G1OpticalDataset
from .expansion_data import ROOT,OUT,read,save,digest

DEST=ROOT/'observation_model/outputs/encoder_v2/G2_arch'
G1=ROOT/'observation_model/outputs/encoder_v2/G1'
TOL=dict(atol=1e-5,rtol=1e-5,sensitivity_min=1e-8,hidden_atol=0.)


def state_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fixed_images(device='cpu'):
    dataset=G1OpticalDataset(OUT,G1,'TRAIN');selected=[];images=[]
    for sensor,count in [('S2',10),('L8',7)]:
        indices=[i for i,r in enumerate(dataset.rows) if r['sensor']==sensor and r['band_count']==count]
        index=min(indices,key=lambda i:sum(np.prod(b['shape']) for b in dataset.rows[i]['bands']))
        row=dataset.rows[index];selected.append(dict(id=row['id'],cache_sha256=row['cache_sha256'],sensor=sensor,role='TRAIN',full_window=True))
        images.append(image_from_g1(dataset[index],device))
    return images,selected


def masked_rectangle(image,rectangle):
    # Test fixture, not the G3 random mask scheduler: remove every intersecting pixel.
    left,bottom,right,top=rectangle;bands=[]
    for band in image.bands:
        gx,gy=band.support;_,_,c,_,_,f=band.transform;h,w=band.values.shape
        xx=torch.arange(w,device=band.values.device,dtype=torch.float64)*gx+c
        yy=f-torch.arange(h,device=band.values.device,dtype=torch.float64)*gy
        hidden=(xx[None]<right)&(xx[None]+gx>left)&(yy[:,None]>bottom)&(yy[:,None]-gy<top)
        bands.append(replace(band,visible=~hidden))
    return OpticalImage(bands,image.product_id)


def cpu_checks():
    started=time.perf_counter();config=EncoderConfig();DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'cpu_acceptance.json').exists():raise FileExistsError('Preserve G2 CPU acceptance')
    assert read(G1/'data_contract_tests.json')['status']=='PASS' and read(G1/'real_product_checks.json')['status']=='PASS'
    torch.manual_seed(0);torch.set_num_threads(4)
    save(DEST/'config.json',dict(**asdict(config),tolerances=TOL,seed=0,optimizer_updates=0,
        G2_authorized=True,G3_authorized=False,G4_authorized=False,
        GPU_profile_budget_seconds=120,GPU_memory_cap_bytes=8*1024**3,
        reference='Fixed capacity128, four spatial blocks; existing AttentionBlock/CrossBlock reused'))
    images,members=fixed_images();save(DEST/'members.json',members)
    model=ObservationEncoderV2(config).eval();initial=state_hash(model)
    torch.manual_seed(0);other=ObservationEncoderV2(config)
    assert initial==state_hash(other);del other
    grids={str(g):token_grid((0,0,1200,1200),g,config) for g in [10,20,30,.5]}
    assert [grids[str(g)]['raw_tokens'] for g in [10,20,30,.5]]==[400,200,133,8000]
    capped=token_grid((0,0,10000,10000),.5,config)
    assert capped['token_budget_capped'] and capped['actual_tokens']<=config.max_tokens
    save(DEST/'shape_tests.json',dict(status='PASS',reference_grids=grids,cap_case=capped))
    curve=np.arange(300.,2601.)
    narrow=np.exp(-.5*((curve-700)/15)**2);narrow/=np.trapezoid(narrow,curve)
    wide=np.exp(-.5*((curve-700)/60)**2);wide/=np.trapezoid(wide,curve)
    a,b=srf_descriptor(curve,narrow),srf_descriptor(curve,wide)
    assert abs(float(a.sum())-1)<1e-6 and abs(float(b.sum())-1)<1e-6
    p=model.encoder.projection
    bandwidth_diff=float((p.generated_weight(a,(10,10))-p.generated_weight(b,(10,10))).abs().max().detach())
    gsd_diff=float((p.generated_weight(a,(10,10))-p.generated_weight(a,(30,30))).abs().max().detach())
    assert bandwidth_diff>1e-8 and gsd_diff>1e-8
    off=ObservationEncoderV2(config,'off');off.load_state_dict(model.state_dict())
    frozen=ObservationEncoderV2(config,'frozen');frozen.load_state_dict(model.state_dict())
    assert sum(p.numel() for p in off.parameters())==sum(p.numel() for p in model.parameters())
    assert torch.equal(off.encoder.projection.generated_weight(a,(10,10)),off.encoder.projection.shared.weight)
    assert not any(p.requires_grad for p in frozen.encoder.projection.hyper.parameters())
    save(DEST/'hypernet_tests.json',dict(status='PASS',same_center_different_bandwidth_weight_max=bandwidth_diff,
        changed_GSD_weight_max=gsd_diff,condition_off_exact_shared_weight=True,frozen_condition_available=True,
        parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        decoder_parameters=sum(p.numel() for p in model.decoder.parameters()),random_initialization=True))
    del off,frozen
    test=OpticalBand(torch.arange(1,5,dtype=torch.float32)[None].expand(4,-1),torch.ones(4,4,dtype=torch.bool),
        (30.,0.,15.,0.,-30.,120.),'EPSG:32650',a,'synthetic')
    geometry=dict(shape=(2,2),bounds=(0.,0.,120.,120.))
    moments,valid,coverage=pool_native(test,geometry,config)
    assert torch.allclose(moments[:,0].reshape(2,2),torch.tensor([[4/3,3.],[4/3,3.]]))
    assert torch.allclose(coverage.reshape(2,2),torch.tensor([[.75,1],[.75,1.]]))
    save(DEST/'geometry_tests.json',dict(status='PASS',source_phase_m=15,expected_area_means=[[4/3,3.],[4/3,3.]],
        observed_area_means=moments[:,0].reshape(2,2).tolist(),coverage=coverage.reshape(2,2).tolist(),
        original_values_not_resampled=True))
    with torch.no_grad():
        rep=model.encode(images)
        separate=[model.encode(image) for image in images]
        for i,r in enumerate(separate):
            assert torch.allclose(rep.features[i,:r.features.shape[1]],r.features[0],**{k:TOL[k] for k in ['atol','rtol']})
            assert not rep.valid[i,r.features.shape[1]:].any()
        reversed_rep=model.encode(OpticalImage(list(reversed(images[0].bands))))
        permutation_error=float((reversed_rep.features-separate[0].features).abs().max())
        assert torch.allclose(reversed_rep.features,separate[0].features,atol=TOL['atol'],rtol=TOL['rtol'])
        shape_outputs=[]
        for i,image in enumerate(images):
            resolutions=[10,20] if i==0 else [30]
            for g in resolutions:
                band=next(b for b in image.bands if b.support==(g,g))
                y,q=model.decoder(rep,target_from_band(band),i)
                assert y.shape==band.values.shape and q.any() and torch.isfinite(y).all()
                assert float(y[q].var())>1e-8
                shape_outputs.append(dict(resolution_m=g,shape=list(y.shape),valid=int(q.sum()),prediction_variance=float(y[q].var())))
        # One synthetic half-meter example: native input is generated, not WorldView.
        synth=OpticalImage([OpticalBand(torch.randn(256,256),torch.ones(256,256,dtype=torch.bool),
            (.5,0.,500000.25,0.,-.5,4000000.25),'EPSG:32650',a,'synthetic_0.5m')])
        fine=model.encode(synth);assert fine.features.shape[1]==fine.grids[0]['actual_tokens']
        first=fine.footprints[0,0];assert float(first[0])==500000.25 and first.dtype==torch.float64
        fine_band=replace(synth.bands[0],values=synth.bands[0].values.clone());fine_band.values[128,128]+=.1
        fine_changed=model.encode(OpticalImage([fine_band]))
        fine_sensitivity=float((fine_changed.features-fine.features).abs().max());assert fine_sensitivity>TOL['sensitivity_min']
        image=images[0];left,bottom,right,top=separate[0].grids[0]['bounds']
        cx=(left+right)/2;cy=(bottom+top)/2
        masked=masked_rectangle(image,(cx-30,cy-30,cx+30,cy+30));target=target_from_band(image.bands[0])
        hidden_rep=model.encode(masked);hidden_y=model.decoder(hidden_rep,target)[0]
        altered=OpticalImage([replace(b,values=torch.where(b.visible,b.values,b.values+10000),
            valid=torch.where(b.visible,b.valid,~b.valid)) for b in masked.bands])
        changed_rep=model.encode(altered);changed_y=model.decoder(changed_rep,target)[0]
        hidden_delta=float((hidden_rep.features-changed_rep.features).abs().max())
        prediction_delta=float((hidden_y-changed_y).abs().max())
        assert hidden_delta==prediction_delta==0
        excluded=image.bands[-1].band_key
        base=model.encode(image,(excluded,));by=model.decoder(base,target_from_band(image.bands[-1]))[0]
        altered=OpticalImage([replace(b,values=b.values+10000,valid=~b.valid,srf=b.srf.flip(0)) if b.band_key==excluded else b for b in image.bands])
        br=model.encode(altered,(excluded,));py=model.decoder(br,target_from_band(image.bands[-1]))[0]
        assert torch.equal(base.features,br.features) and torch.equal(by,py)
        changed=replace(image.bands[0],values=image.bands[0].values.clone())
        yy,xx=torch.nonzero(changed.valid,as_tuple=True);changed.values[yy[len(yy)//2],xx[len(xx)//2]]+=.1
        visible_delta=float((model.encode(OpticalImage([changed]+image.bands[1:])).features-separate[0].features).abs().max())
        assert visible_delta>TOL['sensitivity_min']
        zero=replace(hidden_rep,features=torch.zeros_like(hidden_rep.features))
        decoder_H_delta=float((model.decoder(zero,target)[0]-hidden_y).abs().max());assert decoder_H_delta>TOL['sensitivity_min']
    save(DEST/'leakage_tests.json',dict(status='PASS',A_hidden_value_and_QA_intervention_max=hidden_delta,
        A_prediction_intervention_max=prediction_delta,B_hidden_entire_band_and_SRF_ignored=True,
        visible_local_pixel_sensitivity=visible_delta,synthetic_half_meter_sensitivity=fine_sensitivity,
        decoder_H_sensitivity=decoder_H_delta,decoder_no_raw_value_argument=True,permutation_max_abs=permutation_error,
        synthetic_half_meter_grid=fine.grids[0],real_native_outputs=shape_outputs))
    # Differentiate through a fixed valid native target; never construct an optimizer.
    variable=replace(masked.bands[0],values=masked.bands[0].values.detach().clone().requires_grad_(True))
    grad_image=OpticalImage([variable]+masked.bands[1:]);prediction,q=model.predict(grad_image,target)
    loss=(prediction[q]-images[0].bands[0].values[q]).square().mean();loss.backward()
    assert variable.values.grad[~variable.visible].eq(0).all()
    gradients={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
    assert gradients and all(math_finite(v) for v in gradients.values())
    assert any(v>0 for k,v in gradients.items() if '.hyper.' in k)
    assert any(v>0 for k,v in gradients.items() if k.startswith('decoder.'))
    assert state_hash(model)==initial
    save(DEST/'cpu_acceptance.json',dict(status='PASS',loss=float(loss.detach()),gradient_norms=gradients,
        hidden_input_gradient_exact_zero=True,model_hash_before=initial,model_hash_after=state_hash(model),
        optimizer_updates=0,fixed_seed_state_reproducible=True,wall_seconds=time.perf_counter()-started))
    torch.save(dict(model=model.state_dict(),config=asdict(config),optimizer_updates=0,initialization='seed0_random_G2_untrained'),DEST/'initialization_untrained.pt')
    print('G2_CPU_PASS',time.perf_counter()-started,flush=True)


def math_finite(value):return bool(np.isfinite(value))


def gpu_check(device):
    assert read(DEST/'cpu_acceptance.json')['status']=='PASS'
    if (DEST/'memory_profile.json').exists():raise FileExistsError('One bounded GPU check only; preserve the prior result')
    started=time.perf_counter();result=dict(status='NOT_RUN',optimizer_updates=0,device=device)
    try:
        # One process, one context, one mixed native batch forward/backward.
        torch.set_num_threads(4);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0);model=ObservationEncoderV2().to(device).eval();before=state_hash(model)
        images,members=fixed_images(device)
        torch.cuda.synchronize();t=time.perf_counter();rep=model.encode(images);loss=0
        outputs=[]
        for i,image in enumerate(images):
            band=image.bands[0];pred,valid=model.decoder(rep,target_from_band(band),i)
            q=valid&band.valid;loss=loss+(pred[q]-band.values[q]).square().mean()/len(images)
            outputs.append(dict(shape=list(pred.shape),finite=bool(torch.isfinite(pred).all())))
        torch.cuda.synchronize();forward=time.perf_counter()-t;t=time.perf_counter();loss.backward();torch.cuda.synchronize()
        backward=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        norms={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
        assert torch.isfinite(loss) and all(math_finite(v) for v in norms.values())
        assert any(v>0 for k,v in norms.items() if '.hyper.' in k)
        assert before==state_hash(model) and before==read(DEST/'cpu_acceptance.json')['model_hash_before']
        assert peak<=8*1024**3 and forward+backward<=120
        result.update(status='PASS',forward_seconds=forward,backward_seconds=backward,peak_allocated_bytes=peak,
            peak_reserved_bytes=reserved,loss=float(loss.detach()),outputs=outputs,members=members,
            token_counts=[g['actual_tokens'] for g in rep.grids],gradient_norms=norms,
            model_hash_before=before,model_hash_after=state_hash(model),
            gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,CUDA_version=torch.version.cuda)
    except Exception as exc:
        result.update(status='FAIL_STOP_G2',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-started;save(DEST/'memory_profile.json',result)
    print('G2_GPU_PASS',result['peak_allocated_bytes'],forward,backward,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',default=None);args=parser.parse_args()
    try:
        if args.gpu:gpu_check(args.gpu)
        else:cpu_checks()
    except Exception:
        DEST.mkdir(parents=True,exist_ok=True)
        save(DEST/'failure.json',dict(status='FAIL_STOP_G2',traceback=traceback.format_exc(),optimizer_updates=0))
        raise


if __name__=='__main__':main()
