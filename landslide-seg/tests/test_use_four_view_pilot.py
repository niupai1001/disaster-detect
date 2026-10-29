"""Small offline checks for the download pilot's trust boundaries."""
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


spec=importlib.util.spec_from_file_location('four_view_pilot',Path(__file__).resolve().parents[1]/'scripts/download_use_four_view_pilot.py')
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_quality_does_not_average_class_codes(tmp_path):
    path=tmp_path/'qa.tif'
    values=np.full((72,72),8,dtype='uint8')
    with rasterio.open(path,'w',driver='GTiff',width=72,height=72,count=1,dtype='uint8',
                       crs='EPSG:32631',transform=from_origin(500000,5000000,20,20)) as dst:
        dst.write(values,1)
    with pytest.raises(ValueError,match='quality'):
        pilot.quality(path,'EPSG:32631',[500000,4998560,501440,5000000],'scl')


def test_crop_rejects_partial_coverage(tmp_path,monkeypatch):
    path=tmp_path/'source.tif'
    with rasterio.open(path,'w',driver='GTiff',width=10,height=10,count=1,dtype='uint16',
                       crs='EPSG:32631',transform=from_origin(500000,5000000,10,10)) as dst:
        dst.write(np.ones((10,10),dtype='uint16'),1)
    monkeypatch.setattr(pilot,'DATA',tmp_path)
    with pytest.raises(ValueError,match='not fully covered'):
        pilot.crop_raster(path,tmp_path/'crop.tif','EPSG:32631',[500000,4999800,500200,5000000])


def test_ignored_range_is_rejected_before_body(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'OUT',tmp_path)
    class Response:
        status_code=200
        headers={}
        closed=False
        def raise_for_status(self):pass
        def iter_content(self,*_):
            raise AssertionError('An unbounded archive body must never be read')
        def close(self):self.closed=True
    response=Response()
    network=pilot.Network(limit=100_000_000)
    monkeypatch.setattr(network.session,'request',lambda *a,**k:response)
    with pytest.raises(RuntimeError,match='Range'):
        network.get('https://example.com/archive.zip',start=20,end=35)
    assert response.closed and network.state['received_bytes']==0


def test_budget_rejects_before_request(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'OUT',tmp_path)
    network=pilot.Network(limit=1)
    monkeypatch.setattr(network.session,'request',lambda *a,**k:pytest.fail('Budget already exhausted'))
    with pytest.raises(pilot.BudgetExceeded):
        network.get('https://example.com/data.tif')


def test_parallel_requests_reserve_total_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'OUT',tmp_path)
    network=pilot.Network(limit=300_000)
    ready=threading.Event();release=threading.Event();lock=threading.Lock();calls=0
    class Response:
        status_code=200
        headers={'Content-Length':'80000'}
        def raise_for_status(self):pass
        def iter_content(self,*_):
            assert release.wait(5)
            yield b'x'*80000
        def close(self):pass
    def request(*args,**kwargs):
        nonlocal calls
        with lock:
            calls+=1
            if calls==2:ready.set()
        return Response()
    monkeypatch.setattr(network.session,'request',request)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(network.get,'https://example.com/data',max_bytes=80000) for _ in range(2)]
        try:
            assert ready.wait(5)
            with pytest.raises(pilot.BudgetExceeded):network.get('https://example.com/third',max_bytes=80000)
        finally:release.set()
        assert all(len(f.result()[0])==80000 for f in futures)
    network.flush(True)
    assert calls==2 and network.reserved==0
    assert network.state['received_bytes']==160000<=network.limit


def test_parallel_catalog_cache_fetches_once(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'OUT',tmp_path)
    calls=[]
    def fetch(*args):
        calls.append(1);time.sleep(.02)
        return {'items':[1,2]}
    net=SimpleNamespace(json=fetch)
    with ThreadPoolExecutor(max_workers=6) as pool:
        values=list(pool.map(lambda _:pilot.cached_json(net,'same','https://example.com/catalog'),range(6)))
    assert len(calls)==1 and all(v=={'items':[1,2]} for v in values)


def test_position_scheduler_is_parallel_and_bounded(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'OUT',tmp_path/'out')
    monkeypatch.setattr(pilot,'DATA',tmp_path/'data')
    monkeypatch.setitem(sys.modules,'prepare_use_external',SimpleNamespace(exclusion=lambda:[]))
    class Archive:
        fp=SimpleNamespace(cache=[])
        def infolist(self):return [SimpleNamespace(filename='hr_dataset_raw/root/file.JP2')]
        def close(self):pass
    monkeypatch.setattr(pilot,'RemoteZip',lambda *a,**k:None)
    monkeypatch.setattr(pilot.zipfile,'ZipFile',lambda *_:Archive())
    proxy=SimpleNamespace(reader=SimpleNamespace(shutdown=lambda **k:None),server=SimpleNamespace(shutdown=lambda:None))
    monkeypatch.setattr(pilot,'AssetProxy',lambda *a,**k:proxy)
    active=0;peak=0;lock=threading.Lock();barrier=threading.Barrier(3)
    def process(net,proxy,archive,row,roots):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        barrier.wait(timeout=5)
        time.sleep(.01)
        with lock:active-=1
        return {'site_id':row['tile'],'files':{}}
    monkeypatch.setattr(pilot,'process_site',process)
    frame=SimpleNamespace(drop_duplicates=lambda _:SimpleNamespace(to_dict=lambda _:[{'tile':str(i)} for i in range(8)]))
    net=SimpleNamespace(state={'received_bytes':0},limit=9_680_000_000,flush=lambda **k:None)
    pilot.run(net,Archive(),frame,max_sites=3,site_workers=3,reader_workers=6)
    completed=json.loads((pilot.DATA/'manifest.json').read_text())
    status=json.loads((pilot.OUT/'status.json').read_text())
    assert peak==3 and len(completed)==3
    assert len({x['site_id'] for x in completed})==3 and status['active_sites']==[]


def test_resume_keeps_verified_crops_and_quarantines_uncommitted(tmp_path,monkeypatch):
    monkeypatch.setattr(pilot,'OUT',tmp_path/'out')
    monkeypatch.setattr(pilot,'DATA',tmp_path/'data')
    folder=pilot.DATA/'staging'/'one'/'s2'
    folder.mkdir(parents=True)
    good=folder/'good.tif';good.write_bytes(b'completed')
    bad=folder/'partial.tif';bad.write_bytes(b'partial')
    pilot.save(folder/'product.json',{'assets':{'good':{'file':'staging/one/s2/good.tif','sha256':pilot.hash_file(good)}}})
    pilot.prepare_parallel_resume()
    assert good.read_bytes()==b'completed' and not bad.exists()
    assert bad.with_name('partial.tif.unverified_serial').read_bytes()==b'partial'
