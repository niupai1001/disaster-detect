"""Zero-optimizer-update coverage and local spectral-dependence diagnostics."""
import json
import torch
from pathlib import Path

if __name__=='__main__':torch.cuda.init()

from .data import NativeSet,RELEASE
from .token_prototype import PatchModel
from .token_experiment_a import OUT,select,slice_items,masks,features,fit_linear,summary


def main():
    torch.set_num_threads(4);r=json.loads((OUT/'result.json').read_text())
    if not r['fit_gate']:raise RuntimeError('A fitting gate failed; no held-out diagnosis')
    state=torch.load(OUT/'model.pt',map_location='cuda',weights_only=False);stats=state['statistics']
    train,tm=select(NativeSet('train'),stats,24);val,vm=select(NativeSet('val'),stats,16)
    assert tm+vm==json.loads((OUT/'members.json').read_text())
    fits=fit_linear(train);torch.save(fits,OUT/'spectral_fits.pt')
    coverage={}
    for sensor,(items,targets) in train.items():
        for j,b in enumerate(items):
            x=[]
            for i in (targets==j).nonzero().flatten().tolist():
                value,ok=features(items,j,i);x.append(value[ok.all(-1)])
            x=torch.cat(x);low=x.amin(0);high=x.amax(0);vi,vt=val[sensor];outside=0;count=0
            for i in (vt==j).nonzero().flatten().tolist():
                values,ok=features(vi,j,i);values=values[ok.all(-1)];outside+=int(((values<low)|(values>high)).any(-1).sum());count+=len(values)
            coverage[b.name]=dict(train_target_scenes=int((targets==j).sum()),val_target_scenes=int((vt==j).sum()),train_regression_pixels=fits[b.name]['train_pixels'],val_complete_feature_pixels=count,outside_train_feature_range_fraction=outside/max(count,1))
    model=PatchModel().cuda();model.load_state_dict(state['model']);model.eval();rows=[]
    with torch.no_grad():
        for sensor,(items,targets) in val.items():
            for begin in range(0,len(targets),4):
                ix=torch.arange(begin,min(begin+4,len(targets)),device='cuda');bs=slice_items(items,ix);ts=targets[ix];h=masks(bs,ts)
                h[:,:,2:4,2:4]=True
                pred=model(bs,h)
                for k,j in enumerate(ts.tolist()):
                    b=bs[j];p=60//b.native_gsd_m;valid=b.valid[k,p:2*p,p:2*p];e=(pred[j][k,p:2*p,p:2*p]-b.value[k,p:2*p,p:2*p])[valid]
                    rows.append(dict(sensor=sensor,band=b.name,index=int(ix[k]),mse=float(e.square().mean())))
    result=dict(status='A_ZERO_UPDATE_DIAGNOSTICS_COMPLETE',coverage=coverage,spectral_region_removed=summary(rows),normal_A_score=r['val_model']['score'],optimizer_updates=0,
        boundary='removing other bands at the same location changes the input distribution; dependence evidence only, not proof of correct spectral mapping; identical evaluated target pixels')
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(normal_A=r['val_model']['score'],other_local_spectra_removed=result['spectral_region_removed']['score'])))


if __name__=='__main__':main()
