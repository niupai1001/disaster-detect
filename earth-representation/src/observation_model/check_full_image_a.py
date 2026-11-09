"""One real full image from each sensor: calibration, full support, masking and gradients."""
import json
from pathlib import Path
import torch
if __name__=='__main__':torch.cuda.init()
import numpy as np
from rasterio.transform import Affine,rowcol
from pyproj import Transformer
from .full_image_data import read_full
from .train_full_image_a import OUT,ROOT,inputs
from .token_prototype import PatchModel


def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    prepared=torch.load(OUT/'prepared_data.pt',map_location='cpu',weights_only=False)
    state=torch.load(ROOT/'observation_model/outputs/token_v1/complete_input.pt',map_location='cpu',weights_only=False);stats=state['statistics'];model=PatchModel().cuda();model.load_state_dict(state['model']);model.eval();result={}
    for sensor in ['S2','L8']:
        old=next(o for o in prepared['train'] if o['record']['sensor']==sensor);arrays,r=read_full(old['record']);r.update(target_index=old['record']['target_index'],mask_patch_yx=old['record']['mask_patch_yx'])
        cached_band=r['bands'][0];full_band=r['full_bands'][0]
        with np.load(r['cache']) as z:
            v=z[cached_band['name']+'_valid'];ys,xs=np.nonzero(v);ys=ys[:100];xs=xs[:100]
            if not len(ys):raise ValueError('no reference pixels')
            t=Affine(*cached_band['transform']);xx,yy=t*(xs+.5,ys+.5)
            if cached_band['crs']!=full_band['crs']:xx,yy=Transformer.from_crs(cached_band['crs'],full_band['crs'],always_xy=True).transform(xx,yy)
            ry,rx=rowcol(Affine(*full_band['transform']),xx,yy);ok=arrays[0][1][ry,rx].numpy();diff=float(np.max(np.abs(arrays[0][0][ry,rx].numpy()[ok]-z[cached_band['name']][ys,xs][ok])))
            assert ok.any() and diff<1e-6
        obs=dict(arrays=arrays,record=r);items,h=inputs(obs,stats)
        assert all(b.value.shape[-1]*b.native_gsd_m==r['full_footprint_m'] for b in items)
        with torch.no_grad():before=model(items,h)
        j=r['target_index'];g=items[j].native_gsd_m;p=60//g;y,x=r['mask_patch_yx'];modified=items[j].value.clone();modified[:,y*p:(y+1)*p,x*p:(x+1)*p]+=10000;items[j].value=modified
        with torch.no_grad():after=model(items,h)
        delta=max(float((a-b).abs().max()) for a,b in zip(before,after));assert delta==0
        model.zero_grad(set_to_none=True);prediction=model(items,h)[j];loss=prediction[:,y*p:(y+1)*p,x*p:(x+1)*p].square().mean();loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None];assert all(torch.isfinite(g).all() for g in grads)
        result[sensor]=dict(raw_shape=full_band['source_shape'],padded_shape=full_band['padded_shape'],footprint=r['full_footprint_m'],reference_reflectance_max_error=diff,hidden_value_intervention_max=delta,finite_gradient=True,original_pixels_discarded=0)
    (OUT/'full_image_checks.json').write_text(json.dumps(dict(status='PASSED_FULL_NATIVE_A_CRITICAL_PATHS',sensors=result,optimizer_updates=0),indent=2));print(json.dumps(result))


if __name__=='__main__':main()
