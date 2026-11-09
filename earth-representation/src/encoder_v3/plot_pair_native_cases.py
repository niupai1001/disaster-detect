"""Display native-grid TRAIN reflectance and QA for pair review; no resampling."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import NativeOpticalDataset,sha256
from .geometry_views import common_metric_view


def main():
    reports=Path(__file__).resolve().parent/'reports';study=json.loads((reports/'pilot_study_candidate.json').read_text(encoding='utf-8'));ds=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
    if ds.identities!=study['source_identities']:raise ValueError('Frozen source identities differ')
    pairs=[('obs_57c4c169ec25701452c35149','obs_8b41ee13dedf08054ec49578'),('obs_973cae81321bcaab1ebaebee','obs_06911cf5e63c9c5455a58ae1'),('obs_89dde414bba657df2d0b781c','obs_a11a4b1912c82ef60ee3181a')]
    lookup={x['id']:i for i,x in enumerate(ds.rows)};catalog=json.loads((reports/'cross_source_pair_candidates.json').read_text(encoding='utf-8'));fig,axes=plt.subplots(len(pairs),2,figsize=(11,14),constrained_layout=True);audit=[]
    for row,(a_id,b_id) in enumerate(pairs):
        pair=next(p for p in catalog['pairs'] if p['S2_id']==a_id and p['Landsat_id']==b_id)
        if pair['role']!='TRAIN':raise ValueError('TRAIN visual review only')
        a=ds[lookup[a_id]];b=common_metric_view(ds[lookup[b_id]],a.bands[0].crs);boxes=[]
        for obs in (a,b):
            for band in obs.bands[:1]:
                h,w=band.reflectance.shape;gx,_,left,_,e,top=band.affine;boxes.append((left,top+h*e,left+w*gx,top))
        origin=(min(v[0] for v in boxes),min(v[1] for v in boxes));xmax=max(v[2] for v in boxes)-origin[0];ymax=max(v[3] for v in boxes)-origin[1];entries=[]
        for col,(obs,keys) in enumerate(((a,('B04','B03','B02')),(b,('SR_B4','SR_B3','SR_B2')))):
            bands={x.record_id:x for x in obs.bands};selected=[bands[k] for k in keys]
            if any(x.affine!=selected[0].affine or x.reflectance.shape!=selected[0].reflectance.shape for x in selected):raise ValueError('RGB grids differ; no display resampling fallback')
            raw=np.stack([x.reflectance.numpy() for x in selected],axis=-1);mask=np.logical_and.reduce([x.keep().numpy() for x in selected]);rgb=np.where(mask[...,None],np.clip(raw/.3,0,1),.75)
            x0,y0,x1,y1=boxes[col];ax=axes[row,col];ax.imshow(rgb,origin='upper',extent=(x0-origin[0],x1-origin[0],y0-origin[1],y1-origin[1]),interpolation='nearest');ax.set_xlim(0,xmax);ax.set_ylim(0,ymax);ax.set_aspect('equal');ax.set_title(('S2' if col==0 else 'Landsat')+f" | Pair {row+1}, gap {pair['time_gap_seconds']/3600:.3f} h");ax.set_xlabel('Easting from pair origin (m)');ax.set_ylabel('Northing from pair origin (m)')
            if sha256(obs.metadata['cache'])!=obs.metadata['cache_sha256']:raise ValueError('Cache changed during visual review')
            entries.append(dict(id=obs.identifier,cache_sha256=obs.metadata['cache_sha256'],product=obs.metadata['product'],native_shape=list(raw.shape[:2]),display_bands=keys,RGB_all_valid_pixels=int(mask.sum()),original_native_affine=selected[0].affine))
        audit.append(dict(pair=pair,origin_xy=origin,members=entries))
    fig.suptitle('Native RGB, same physical frame | gray = invalid QA | display scale 0-0.3 reflectance',fontsize=12)
    output=reports/'pair_native_review_cases.png';fig.savefig(output,dpi=130);plt.close(fig)
    (reports/'pair_native_review_cases.json').write_text(json.dumps(dict(status='NATIVE_TRAIN_VISUAL_REVIEW_NOT_ADMISSION',source_identities=ds.identities,rows=audit,imagery_resampled=False,model_updates=0,display_only=dict(reflectance_range=[0,.3],invalid_gray=.75),source_code_sha256=sha256(Path(__file__)),figure_sha256=sha256(output),limits=['Visual appearance does not establish no surface change','Display clipping affects figure only; source values and training masks unchanged','No geolocation correction applied']),indent=2)+'\n',encoding='utf-8');print(output)


if __name__=='__main__':main()
