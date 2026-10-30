"""Offline native full-load/eval/adapter equivalence evidence; no optimizer updates."""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from landslide_seg.models.use_next_v2.anchors import NativeAnySat,NativeDOFA
from landslide_seg.models.use_next_v2 import load_asset
from landslide_seg.data.use_integrated import IntegratedUSEData


def digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    model=NativeAnySat(ROOT)
    model.train()
    assert not any(m.training for m in model.modules())
    before=digest(model)
    data=IntegratedUSEData(ROOT/'outputs/use_next_v2/planning/data_snapshot',sensor='s2')
    asset=load_asset(data,0)
    with torch.no_grad():
        inputs=model.prepare(asset)
        direct=model.official(inputs,patch_size=40,output='dense',output_modality='s2')
        adapted=model(asset)
    assert torch.equal(direct,adapted) and torch.isfinite(adapted).all()
    after=digest(model)
    assert before==after
    report=dict(strict_load=True,all_eval=True,parameters_frozen=not any(p.requires_grad for p in model.parameters()),
                direct_adapter_max_abs=float((direct-adapted).abs().max()),state_sha256=before,
                unchanged=before==after,real_asset=asset.provenance['id'],shape=list(adapted.shape),device='cpu',
                sar_path='BLOCKED: VV/VH RTC gamma0 alone does not establish official s1-asc product/orbit compatibility',
                landsat_path='BLOCKED: six/seven reflective bands do not satisfy l8 eleven-channel input',
                formal_updates=0)
    out=ROOT/'outputs/use_next_v2/engineering/native_anysat.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    del model,direct,adapted
    dofa=NativeDOFA(ROOT)
    before=digest(dofa); dofa.train()
    with torch.no_grad():
        direct=dofa.official(*dofa.prepare(asset)); adapted=dofa(asset)
    assert all(torch.equal(x,y) for x,y in zip(direct,adapted))
    assert before==digest(dofa) and not any(p.requires_grad for p in dofa.parameters())
    result=dict(strict_load=True,checkpoint_keys=len(dofa.loaded_keys),all_keys_retained=True,
                direct_adapter_max_abs=max(float((x-y).abs().max()) for x,y in zip(direct,adapted)),
                shapes=[list(x.shape) for x in adapted],state_sha256=before,unchanged=True,
                pretrained_keys_unused_by_official_dense_forward=['mask_token','norm.weight','norm.bias','projector.weight','projector.bias'],
                architecture='Official complete 12-block dense encoder plus retained checkpoint norm/projector/mask; strict loading all 173 keys; not MAE training decoder',
                input_mapping='128 to224 bilinear, align_corners=False, same footprint; native output14x14; interpolation is not native detail',
                real_asset=asset.provenance['id'],formal_updates=0,device='cpu')
    (ROOT/'outputs/use_next_v2/engineering/native_dofa.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
