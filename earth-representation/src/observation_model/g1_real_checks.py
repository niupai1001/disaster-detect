"""Three real native-product checks, without modifying the released arrays."""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from .expansion_data import ROOT, OUT, lines, read, save, digest, BAD
from .g1_srf_data import DEST
from .native_data import aerosol_valid


def main():
    assert read(DEST/'data_contract_tests.json')['status']=='PASS'
    rows=lines(OUT/'members.jsonl');original={r['id']:r for r in lines(OUT/'canonical_input.jsonl')}
    results=[]
    for sensor,resolution in [('S2',10),('S2',20),('L8',30)]:
        row=next(r for r in rows if r['sensor']==sensor and r['role']=='TRAIN'
            and r['provenance']=='existing_native_TIF_with_QA'
            and any(b['native_pixel_support_m']==[resolution,resolution] for b in r['bands']))
        band=next(b for b in row['bands'] if b['native_pixel_support_m']==[resolution,resolution])
        with rasterio.open(band['source']) as src:
            dn=src.read(1);valid=(src.dataset_mask()>0)&(dn!=0)&np.isfinite(dn)
            t,c=src.transform,src.crs
        quality_sources=original[row['id']]['quality_paths']
        for path in quality_sources:
            with rasterio.open(path) as src:
                a=src.read(1);q=src.dataset_mask()>0;qt,qc=src.transform,src.crs
            if sensor=='S2':q &= ~np.isin(a,BAD)
            elif 'qa_pixel' in path:q &= (a.astype('uint16')&63)==0
            elif 'qa_radsat' in path:q &= a==0
            elif 'qa_aerosol' in path:q &= aerosol_valid(a)
            else:raise ValueError('Unrecognized QA product')
            mask=np.zeros(dn.shape,'uint8')
            reproject(q.astype('uint8'),mask,src_transform=qt,src_crs=qc,dst_transform=t,dst_crs=c,resampling=Resampling.nearest)
            valid &= mask.astype(bool)
        with np.load(row['cache'],allow_pickle=False) as z:
            x=z[band['band_key']];actual=z[band['band_key']+'__valid']
        expected=dn.astype('float32')*band['scale']+band['offset']
        error=float(np.max(np.abs(x[valid]-expected[valid])))
        assert np.array_equal(valid,actual) and error<1e-6
        assert list(t)[:6]==band['transform'] and str(c)==band['crs'] and list(dn.shape)==band['shape']
        results.append(dict(id=row['id'],sensor=sensor,resolution_m=resolution,band=band['band_key'],
            source=band['source'],valid_pixels=int(valid.sum()),valid_fraction=float(valid.mean()),
            max_abs_reflectance_error=error,QA_mask_exact=True,native_transform_exact=True))
    result=dict(status='PASS',controls=results,scope='S2 native10m/20m and Landsat30m original DN+QA to released input',
        optimizer_updates=0,new_download_bytes=0)
    save(DEST/'real_product_checks.json',result)
    sources=read(DEST/'source_manifest.json');sources['observation_model/g1_real_checks.py']=digest(ROOT/'observation_model/g1_real_checks.py')
    save(DEST/'source_manifest.json',sources)
    print(result)


if __name__=='__main__':main()
