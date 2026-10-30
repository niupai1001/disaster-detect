"""Real combined P0 forward/backward, frozen/EMA and exact CUDA disk replay."""
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from landslide_seg.models.use_next_v2 import load_asset
from landslide_seg.models.use_next_v2.physical import area_grid
from landslide_seg.models.use_next_v2.pipeline import USEModel
from landslide_seg.models.use_next_v2.readout import state_digest
from landslide_seg.models.use_next_v2.geometry import to_label_grid
from landslide_seg.data.use_integrated import IntegratedUSEData
from device import claim


def main():
    device=sys.argv[1] if len(sys.argv)>1 else 'cuda:0'
    out=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'outputs/use_next_v2/engineering'/('pipeline_'+device.replace(':','_')+'_v2')
    out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(1234);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    with claim(device) as uuid:
        release=ROOT/'outputs/use_next_v2/data/data_release_v1'
        data=IntegratedUSEData(release,sensor='s2');asset=load_asset(data,0)
        asset.image=asset.image.to(device);asset.pixel_valid=asset.pixel_valid.to(device)
        asset.channel_metadata=asset.channel_metadata.to(device);asset.metadata_known=asset.metadata_known.to(device)
        matrix,shape,t,crs=area_grid(asset)
        grid=dict(crs=crs,shape=shape,origin=(t[2],t[5]))
        rng=np.random.default_rng(1234)
        hidden=torch.from_numpy(rng.random((shape[0]//4,shape[1]//4))<.4).repeat_interleave(4,0).repeat_interleave(4,1)
        model=USEModel(ROOT).to(device);model.train()
        opt=torch.optim.AdamW(model.local.parameters(),lr=.0003,weight_decay=.01)
        anchors=(state_digest(model.dofa),state_digest(model.anysat))
        teacher=model(asset,grid,teacher=True)
        target=teacher.global_embedding
        assert not target.requires_grad
        variance=float(teacher.metric_patch.tensor[:,teacher.metric_patch.valid_mask].var())
        assert variance>1e-8
        targets=teacher.metric_patch.tensor[:,teacher.metric_patch.valid_mask].T
        centered=targets-targets.mean(0)
        covariance=centered.T@centered/max(1,len(targets)-1)
        spatial_variance=float(centered.square().mean())
        assert spatial_variance>1e-8
        def step():
            opt.zero_grad();student=model(asset,grid,hidden)
            loss=(student.global_embedding-target).square().mean();loss.backward()
            gradients=[p.grad for p in model.local.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients) and sum(float(g.abs().sum()) for g in gradients)>0
            opt.step();model.ema(.996)
            return loss.detach()
        torch.cuda.reset_peak_memory_stats(device)
        first=step()
        saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),cursor=1)
        torch.save(saved,out/'resume.pt')
        expected=step();expected_digest=state_digest(model);expected_opt=copy.deepcopy(opt.state_dict())
        saved=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        actual=step()
        assert torch.equal(expected,actual) and state_digest(model)==expected_digest
        for k,v in expected_opt['state'].items():
            for field,value in v.items():assert torch.equal(value.cpu(),opt.state_dict()['state'][k][field].cpu())
        assert anchors==(state_digest(model.dofa),state_digest(model.anysat))
        full=model(asset,grid)
        mapped,valid=to_label_grid(full.metric_patch,asset.sample_grid['transform'],asset.sample_grid['crs'],asset.image.shape[-2:])
        assert mapped.shape[-2:]==asset.image.shape[-2:] and valid.any() and torch.isfinite(mapped).all()
        from landslide_seg.models.use_next_v2.masking import mask_asset
        _,pixels=mask_asset(asset,hidden,grid)
        changed=copy.deepcopy(asset);changed.image[:,pixels.to(device)]=100
        original_student=model(asset,grid,hidden);changed_student=model(changed,grid,hidden)
        assert torch.equal(original_student.global_embedding,changed_student.global_embedding)
        assert all(torch.equal(original_student.semantic_levels[k].tensor,changed_student.semantic_levels[k].tensor)
                   for k in original_student.semantic_levels)
        capacities=[]
        for size in (128,256):
            trial=copy.deepcopy(asset)
            trial.image=torch.rand(10,size,size,device=device);trial.pixel_valid=torch.ones_like(trial.image,dtype=torch.bool)
            # Synthetic legal 10m grid is for capacity only; never enters the data release.
            trial.provenance={'id':'SYNTHETIC_CAPACITY_'+str(size)}
            mat,sh,tr,cr=area_grid(trial);g=dict(crs=cr,shape=sh,origin=(tr[2],tr[5]))
            torch.cuda.reset_peak_memory_stats(device);opt.zero_grad()
            output=model(trial,g);output.global_embedding.square().mean().backward()
            capacities.append(dict(size=size,synthetic=True,ground_extent_m=size*10,peak_bytes=torch.cuda.max_memory_allocated(device)))
        sar=load_asset(IntegratedUSEData(release,sensor='s1'),0)
        for key in ('image','pixel_valid','channel_metadata','metadata_known'):setattr(sar,key,getattr(sar,key).to(device))
        mat,sh,tr,cr=area_grid(sar);sarout=model(sar,dict(crs=cr,shape=sh,origin=(tr[2],tr[5])))
        assert sarout.global_embedding is not None and sarout.optical_semantic is None
        report=dict(uuid=uuid,device=device,real_asset=asset.provenance['id'],losses=[float(first),float(expected),float(actual)],
                    exact_disk_replay=True,optimizer_exact=True,cursor=2,successful_engineering_updates=3,formal_updates=0,
                    native_hashes_unchanged=True,teacher_target_detached=True,teacher_target_variance=variance,
                    label_mapping_shape=list(mapped.shape),peak_bytes=torch.cuda.max_memory_allocated(device),
                    semantic_paths=list(full.semantic_levels),shared_mask_120m=True,hidden_input_perturbation_all_paths_exact=True,
                    capacities=capacities,real_sar_only_pass=True,teacher_spatial_variance=spatial_variance,
                    teacher_covariance_trace=float(covariance.trace()))
        (out/'acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
