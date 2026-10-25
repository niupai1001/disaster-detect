import importlib.util
from pathlib import Path
import torch
from landslide_seg.data.optical_observation import OpticalObservation, collate_optical


def test_frozen_external_snapshot_checked_before_any_audit_or_download(tmp_path,monkeypatch):
    import json
    import hashlib
    import pytest
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('frozen_external',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'OUT',tmp_path)
    monkeypatch.setattr(module,'exclusion',lambda:pytest.fail('Changed snapshot reached spatial audit'))
    manifest=tmp_path/'current_manifest.json'
    manifest.write_text(json.dumps([{'sensor':'s2'}]*32+[{'sensor':'landsat'}]*32))
    paused=tmp_path/'acquisition_paused.json'
    paused.write_text(json.dumps({'sensors':{'s2':32,'landsat':32},'manifest_sha256':'changed'}))
    with pytest.raises(ValueError,match='Frozen current dataset changed'):module.audit_pilot(current=True)
    with pytest.raises(RuntimeError,match='Downloads paused by user'):module.main()
    paused.write_text(json.dumps({'sensors':{'s2':5000,'landsat':5000},'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError,match='Frozen current dataset changed'):module.audit_pilot(current=True)


def test_local_unannotated_quality_date_and_event_isolation(tmp_path,monkeypatch):
    import h5py
    import numpy as np
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    path=root/'scripts/prepare_use_local.py'
    spec=importlib.util.spec_from_file_location('prepare_local',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'train_s2_1.nc'
    with h5py.File(source,'w') as f:
        f.attrs['annotated']='False';f.attrs['crs']='EPSG:32632'
        f.create_dataset('spatial_ref',data=0).attrs['GeoTransform']='500000 10 0 5000000 0 -10'
        f.create_dataset('SCL',data=np.stack([np.full((128,128),9),np.full((128,128),4)]))
        f.create_dataset('time',data=[0,30]).attrs['units']='days since 2020-01-01'
        for b in module.BANDS:f.create_dataset(b,data=np.full((2,128,128),2000,dtype='int16'))
        f.create_dataset('MASK',data=np.full((2,128,128),np.nan))
    assert module.read_unannotated(source,{'different_event'},[])[1]=='outside_train_events'
    (arrays,row),reason=module.read_unannotated(source,{'train'},[])
    assert reason is None and row['time_index']==1 and row['date'].startswith('2020-01-31')
    assert np.allclose(arrays['x'],.2) and arrays['pixel_valid'].all()
    assert not arrays['known'][:,1].any() and not arrays['metadata'][:,1].any()
    assert module.read_unannotated(source,{'train'},[module.box(*row['bbox'])])[1]=='withheld_event_buffer'
    with h5py.File(source,'a') as f:f.attrs['annotated']='True'
    assert module.read_unannotated(source,{'train'},[])[1]=='not_explicitly_unannotated'


def test_joint_sampling_keeps_sensor_balance_and_restores_next_batch():
    import numpy as np
    path=Path(__file__).resolve().parents[1]/'scripts/pretrain_use.py'
    spec=importlib.util.spec_from_file_location('joint_pretrain',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    train={key:[{'origin':key,'id':i} for i in range(13)] for key in ('s2','local_s2','landsat')}
    rng=np.random.default_rng(1234);first=module.sample_training(train,rng)
    assert [r['origin'] for r in first]==['s2']*2+['local_s2']*2+['landsat']*4
    saved=rng.bit_generator.state;expected=module.sample_training(train,rng)
    restored=np.random.default_rng();restored.bit_generator.state=saved
    assert module.sample_training(train,restored)==expected


def test_local_spatial_duplicates_coalesced_across_event_names(monkeypatch):
    root=Path(__file__).resolve().parents[1];monkeypatch.syspath_prepend(str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('local_locations',root/'scripts/prepare_use_local.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(file=name,geometry=module.mapping(module.box(*bounds))) for name,bounds in
        [('event1.npz',(0,0,1,1)),('event2.npz',(0,0,1,1)),('adjacent.npz',(1,0,2,1))]]
    kept,rejected=module.unique_locations(rows)
    again,_=module.unique_locations(list(reversed(rows)))
    assert len(kept)==2 and len(rejected)==1 and 'adjacent.npz' in {r['file'] for r in kept}
    assert {r['file'] for r in kept}=={r['file'] for r in again}


def test_preflight_uses_actual_frozen_pretraining_manifest_and_rejects_changes(tmp_path):
    import json
    import pytest
    path=Path(__file__).resolve().parents[1]/'scripts/check_research_protocol.py'
    spec=importlib.util.spec_from_file_location('frozen_preflight',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    external=tmp_path/'outputs/use/external';external.mkdir(parents=True)
    training=tmp_path/'outputs/use/pretraining';training.mkdir()
    manifest=external/'current_manifest.json';manifest.write_text('["actual frozen dataset"]')
    (external/'manifest.json').write_text('["old pilot"]')
    digest=module.sha(manifest)
    (external/'acquisition_paused.json').write_text(json.dumps({'manifest_sha256':digest}))
    (external/'current_acceptance.json').write_text(json.dumps({'ok':True,'manifest_sha256':digest}))
    config=training/'resolved_config.json';config.write_text(json.dumps({'dataset_mode':'frozen_current','manifest_sha256':digest}))
    result=module.pretraining_sources(tmp_path,{'manifest_sha256':digest})
    assert str(manifest) in result and str(external/'manifest.json') not in result
    manifest.write_text('["changed dataset"]')
    with pytest.raises(ValueError,match='预训练成员清单发生改变'):module.pretraining_sources(tmp_path,{'manifest_sha256':digest})


def test_masked_target_head_receives_only_visible_observation():
    path=Path(__file__).resolve().parents[1]/'scripts/pretrain_use.py'
    spec=importlib.util.spec_from_file_location('pretrain_use',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.Reconstruction()
    visible=OpticalObservation(torch.randn(3,7,9),('blue','red','nir'),torch.ones(3,3),
        torch.ones(3,3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),torch.ones(3,7,9,dtype=torch.bool),{})
    batch=collate_optical([visible])
    target_metadata=torch.ones(1,2,3);known=torch.ones(1,2,3,dtype=torch.bool)
    prediction=model(batch,target_metadata,known)
    assert prediction.shape==(1,2,7,9)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_reconstruction_weights_bands_and_samples_equally_and_ignores_missing():
    path=Path(__file__).resolve().parents[1]/'scripts/pretrain_use.py'
    spec=importlib.util.spec_from_file_location('pretrain_loss',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    prediction=torch.tensor([[[[1.,1.]],[[3.,99.]]],[[[5.,99.]],[[99.,99.]]]],requires_grad=True)
    valid=torch.tensor([[[[True,True]],[[True,False]]],[[[True,False]],[[False,False]]]])
    target=torch.zeros_like(prediction);target[~valid]=float('nan')
    loss=module.masked_l1(prediction,target,valid)
    assert loss.item()==3.5  # sample means: (1+3)/2 and 5, then equal mean.
    loss.backward();assert (prediction.grad[~valid]==0).all() and torch.isfinite(prediction.grad).all()


def test_completion_status_retries_transient_snapshot_but_does_not_hide_corruption(monkeypatch):
    import json
    import pytest
    path=Path(__file__).resolve().parents[1]/'scripts/run_use.py'
    spec=importlib.util.spec_from_file_location('use_queue',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    class Status:
        def __init__(self,broken=False):self.calls=0;self.broken=broken
        def read_text(self,**kwargs):
            self.calls+=1
            return '{' if self.broken or self.calls==1 else '{"status":"running"}'
    transient=Status();assert module.read_status(transient)=={'status':'running'} and transient.calls==2
    broken=Status(True)
    with pytest.raises(json.JSONDecodeError):module.read_status(broken)
    assert broken.calls==10


def test_external_signing_cached_and_throttle_stops(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import pytest
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('prepare_external',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    def token(url,**kwargs):
        calls.append(url);return {'token':'public-read-token','msft:expiry':'2100-01-01T00:00:00Z'}
    real_get=module.get;monkeypatch.setattr(module,'get',token)
    urls=['https://account.blob.core.windows.net/container/'+str(i) for i in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(module.signed,urls))
    assert len(calls)==1 and all(u.endswith('?public-read-token') for u in result)
    class Response:
        status_code=429
        headers={'Retry-After':'600'}
    monkeypatch.setattr(module.SESSION,'get',lambda *a,**kw:Response())
    with pytest.raises(module.RemoteThrottled,match='600'):
        real_get('https://example.invalid')
    calls.clear()
    def blocked(url,**kwargs):
        calls.append(url);raise module.RemoteThrottled('429',600)
    monkeypatch.setattr(module,'get',blocked)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(module.signed,'https://blocked.blob.core.windows.net/container/a') for _ in range(4)]
        for future in futures:
            with pytest.raises(module.RemoteThrottled):future.result()
    assert len(calls)==1
    with pytest.raises(ValueError):module.signed('file:///private/data')


def test_catalog_sensor_filter_and_cell_forwarding(monkeypatch,tmp_path):
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('external_scan',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'OUT',tmp_path)
    items=[{'id':'old','properties':{'platform':'landsat-7'}},{'id':'new','properties':{'platform':'landsat-8'}}]
    monkeypatch.setattr(module,'get',lambda *a,**kw:{'features':items})
    seen=[]
    def window(item,sensor,forbidden,existing,cell,slot):
        seen.append((item['id'],cell));return 'window'
    monkeypatch.setattr(module,'read_window',window)
    item,result=module.scan_cell('landsat','landsat-c2-l2',(0,5),[],[])
    assert item['id']=='new' and result=='window' and seen==[('new',(0,5))]


def test_access_token_cache_uses_actual_expiry_and_rejects_stale_provider_cache(monkeypatch):
    from datetime import datetime,timezone
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('external_expiry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    clock=[1000.];monkeypatch.setattr(module.time,'time',lambda:clock[0]);calls=[]
    def response(token,expiry):return {'token':token,'msft:expiry':datetime.fromtimestamp(expiry,timezone.utc).isoformat()}
    replies=iter([response('first',1400),response('stale',1100),response('fresh',2200)])
    def get(url,**kwargs):calls.append(kwargs);return next(replies)
    monkeypatch.setattr(module,'get',get)
    url='https://account.blob.core.windows.net/container/file.tif'
    assert module.signed(url).endswith('?first')
    clock[0]=1100;assert module.signed(url).endswith('?first') and len(calls)==1
    clock[0]=1300;assert module.signed(url).endswith('?fresh') and len(calls)==3
    assert '_ts' in calls[-1]['params']


def test_prefetch_is_bounded_and_preserves_four_cursor_commit_groups(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from shapely.geometry import box
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('external_prefetch',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    seen=[];calls=[]
    def scan(sensor,collection,cell,forbidden,snapshot,slot):
        calls.append(cell);return (cell,slot)
    monkeypatch.setattr(module,'scan_cell',scan)
    cells=[(i,0) for i in range(12)]
    with ThreadPoolExecutor(max_workers=8) as workers:
        iterator=module.prefetched_groups(workers,'landsat','collection',cells,0,12,[],seen)
        cursors,futures,snapshot=next(iterator)
        assert cursors==[0,1,2,3] and snapshot is None
        assert [future.result()[0] for future in futures]==cells[:4]
        seen.append(box(0,0,1,1))
        remaining=list(iterator)
        assert [group[0] for group in remaining]==[[4,5,6,7],[8,9,10,11]]
        assert all(group[2] is not None for group in remaining)
    assert len(calls)==12 and set(calls)==set(cells)


def test_acquisition_errors_do_not_persist_signed_access_queries():
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('external_error_log',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.safe_error("Read failed: 'https://account.blob.core.windows.net/container/a.tif?sig=secret&se=expiry'") == "Read failed: 'https://account.blob.core.windows.net/container/a.tif?[redacted]'"


def test_real_raster_pipeline_scale_quality_and_grid(monkeypatch,tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('external_raster',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    profile=dict(driver='GTiff',width=1200,height=1200,count=1,dtype='uint16',crs='EPSG:32632',
                 transform=from_origin(500000,5000000,30,30),nodata=0)
    sr=tmp_path/'reflectance.tif';qa=tmp_path/'quality.tif';sat=tmp_path/'saturation.tif'
    with rasterio.open(sr,'w',**profile) as dst:dst.write(np.full((1200,1200),18639,dtype='uint16'),1)
    profile['nodata']=65535
    quality=np.zeros((1200,1200),dtype='uint16');quality[600:610,600:610]=8
    with rasterio.open(qa,'w',**profile) as dst:dst.write(quality,1)
    with rasterio.open(sat,'w',**profile) as dst:dst.write(np.zeros_like(quality),1)
    asset=dict(href=str(sr),**{'raster:bands':[dict(scale=.0000275,offset=-.2,spatial_resolution=30)],
        'eo:bands':[dict(center_wavelength=.5,full_width_half_max=.06)]})
    assets={key:asset for key in module.LS};assets.update(qa_pixel=dict(href=str(qa)),qa_radsat=dict(href=str(sat)))
    monkeypatch.setattr(module,'signed',lambda url:url)
    result=module.read_window(dict(id='test',assets=assets,properties={'datetime':'2024-01-01T00:00:00Z'}),'landsat',[],())
    x,valid,metadata,known,provenance,_=result
    assert x.shape==(6,128,128) and valid.shape==x.shape
    np.testing.assert_allclose(x[valid],.3125725,atol=1e-6)
    assert valid.sum()==6*(128*128-100)
    assert provenance['transform'][0]==30 and (metadata[:,2]==30).all() and known.all()
    shifted=module.read_window(dict(id='test',assets=assets,properties={'datetime':'2024-01-01T00:00:00Z'}),
        'landsat',[],(result[-1],),slot=1)
    assert shifted is not None and not shifted[-1].intersects(result[-1])
    for index,expected in ((2,module.CROP_OFFSETS[1][0]),(5,module.CROP_OFFSETS[1][1])):
        assert abs(shifted[4]['transform'][index]-provenance['transform'][index]-expected)<=15
    with rasterio.open(sat,'w',**profile) as dst:dst.write(np.ones_like(quality),1)
    def unnecessary_band_pool(*args,**kwargs):raise AssertionError('Rejected saturation QA must not download optical bands')
    monkeypatch.setattr(module,'ThreadPoolExecutor',unnecessary_band_pool)
    assert module.read_window(dict(id='test',assets=assets,properties={'datetime':'2024-01-01T00:00:00Z'}),
        'landsat',[],()) is None
    asset.update({'proj:shape':[1200,1200],'proj:transform':list(profile['transform'])[:6]})
    def unexpected_read(*args,**kwargs):raise AssertionError('Out-of-cell metadata should avoid raster downloads')
    monkeypatch.setattr(module.rasterio,'open',unexpected_read)
    assert module.read_window(dict(id='test',assets=assets,properties={'datetime':'2024-01-01T00:00:00Z','proj:epsg':32632}),
        'landsat',[],(),cell=(0,5)) is None
    tree=module.STRtree([result[-1]])
    assert module.overlaps(result[-1],tree) and not module.overlaps(shifted[-1],tree)
