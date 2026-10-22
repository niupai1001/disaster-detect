"""Targeted per-frame missingness sidecar; preserves the original X15 cache."""
import concurrent.futures, hashlib, json, sys
from pathlib import Path
import numpy as np
import xarray as xr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from landslide_seg.data.segformer_dataset import S2_10_BANDS
sys.stdout.reconfigure(encoding='utf-8')
OUT=ROOT/'data/s2_temporal_v6_quality'

def inspect(r):
    with xr.open_dataset(ROOT/'data/sen12'/f"{r['sample_id']}.nc",engine='h5netcdf') as ds:
        ds=ds.sortby('time')
        scl=np.asarray(ds.SCL.values)
        bad=(scl==0)
        nf=0
        for b in S2_10_BANDS:
            a=np.asarray(ds[b].values)
            nonfinite=~np.isfinite(a); nf+=int(nonfinite.sum()); bad|=nonfinite
            fv=ds[b].encoding.get('_FillValue',ds[b].attrs.get('_FillValue'))
            if fv is not None and np.isfinite(fv): bad|=(a==fv)
        dem=np.asarray(ds.DEM.values)
        if dem.ndim==3: dem=dem[0]
        dem_bad=~np.isfinite(dem)
        dates=ds.time.values.astype('datetime64[D]')
        assert len(dates)==15 and np.all(dates[1:]>=dates[:-1])
        days=(dates-np.datetime64('2015-12-03')).astype(np.int32)
    return r['row'],bad,dem_bad,days,{'sample_id':r['sample_id'],'nonfinite_band_values':nf,
        'nodata_pixel_fraction':bad.mean(axis=(1,2)).tolist(),
        'cloud_shadow_snow_fraction':np.isin(scl,[3,8,9,10,11]).mean(axis=(1,2)).tolist(),
        'unknown_scl_fraction':(scl==255).mean(axis=(1,2)).tolist()}

def main():
    if (OUT/'complete.json').exists():
        print('Existing completed sidecar retained');return
    OUT.mkdir(parents=True,exist_ok=True)
    src=ROOT/'data/s2_temporal_v5/index.json'
    idx=json.loads(src.read_text(encoding='utf-8')); rows=idx['rows']; n=len(rows)
    assert not idx['bad'] and n==4715
    bad=np.lib.format.open_memmap(OUT/'nodata.npy',mode='w+',dtype=bool,shape=(n,15,128,128))
    dem=np.lib.format.open_memmap(OUT/'dem_nodata.npy',mode='w+',dtype=bool,shape=(n,128,128))
    dates=np.zeros((n,15),dtype=np.int32); quality=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for i,(j,b,d,t,q) in enumerate(ex.map(inspect,rows),1):
            bad[j]=b;dem[j]=d;dates[j]=t;quality.append(q)
            if i%250==0: print(f'[quality] {i}/{n}',flush=True)
    bad.flush();dem.flush();np.save(OUT/'dates.npy',dates)
    (OUT/'per_frame_quality.json').write_text(json.dumps(quality,ensure_ascii=False),encoding='utf-8')
    meta={'n':n,'bands':list(S2_10_BANDS),'index_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
          'policy':'SCL=0 OR source nonfinite OR explicit finite fillvalue; fill with TRAIN mean after normalization; clouds and SCL255 recorded, not silently filled',
          'date_reference':'2015-12-03','source_nonfinite_band_values':sum(x['nonfinite_band_values'] for x in quality),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'complete.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
