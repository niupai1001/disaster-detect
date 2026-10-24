import importlib.util
from pathlib import Path
import torch
from landslide_seg.data.optical_observation import OpticalObservation, collate_optical


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


def test_external_signing_cached_and_throttle_stops(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import pytest
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_use_external.py'
    spec=importlib.util.spec_from_file_location('prepare_external',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    def token(url,**kwargs):
        calls.append(url);return {'token':'public-read-token'}
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
    asset.update({'proj:shape':[1200,1200],'proj:transform':list(profile['transform'])[:6]})
    def unexpected_read(*args,**kwargs):raise AssertionError('Out-of-cell metadata should avoid raster downloads')
    monkeypatch.setattr(module.rasterio,'open',unexpected_read)
    assert module.read_window(dict(id='test',assets=assets,properties={'datetime':'2024-01-01T00:00:00Z','proj:epsg':32632}),
        'landsat',[],(),cell=(0,5)) is None
    tree=module.STRtree([result[-1]])
    assert module.overlaps(result[-1],tree) and not module.overlaps(shifted[-1],tree)
