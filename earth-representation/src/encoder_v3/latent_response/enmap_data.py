"""Native EnMAP L2A QA/calibration release, independent of the frozen optical release."""
import json
import copy
from collections import OrderedDict
from pathlib import Path
import numpy as np
import rasterio
import torch
from encoder_v3.data import Observation,checked_file,sha256
from encoder_v3.model import BandObservation
from encoder_v3.enmap_admission import spectral_metadata
from .registry import ChannelCatalog
from .types import SensorKey
from .data import LRDataset

KEY=SensorKey('EnMAP','HSI')
QA_POLICY='classes in {1,2}; cloud/shadow/haze/cirrus/snow=0; testflags=0; band pixelmask=0; finite and not nodata'


def checked_raster(path,reference):
    ds=rasterio.open(path)
    if ds.crs!=reference.crs or ds.transform!=reference.transform or ds.shape!=reference.shape:
        ds.close();raise ValueError('EnMAP QA geometry differs from spectral product')
    return ds


def build_release(root,output,existing_members):
    if (output/'release.json').exists():raise ValueError('New release directory required; no overwrite of frozen admission')
    output.mkdir(parents=True,exist_ok=True)
    rows=[];sums=np.zeros(224);squares=np.zeros(224);counts=np.zeros(224,dtype=np.int64)
    for image in sorted(root.rglob('*SPECTRAL_IMAGE*.TIF')):
        prefix=image.name.split('-SPECTRAL_IMAGE')[0]
        metadata=list(root.rglob(prefix+'-METADATA.XML'))
        if not metadata:raise ValueError('Missing matching XML')
        bands=spectral_metadata(metadata[0]);assert len(bands)==224
        companions={}
        for key in ('CLASSES','CLOUD','CLOUDSHADOW','HAZE','CIRRUS','SNOW','TESTFLAGS'):
            choices=[p for p in root.rglob(prefix+'-QL_QUALITY_'+key+'*.TIF') if p.name in (prefix+'-QL_QUALITY_'+key+'.TIF',prefix+'-QL_QUALITY_'+key+'_COG.TIF')]
            if len(choices)!=1:raise ValueError('Unique original QA companion required: '+key)
            companions[key]=choices[0]
        masks=list(root.rglob(prefix+'-QL_PIXELMASK*.TIF'))
        if len(masks)!=1:raise ValueError('Original per-band defective pixel mask required')
        companions['PIXELMASK']=masks[0]
        assets={k:dict(path=str(p.resolve()),sha256=sha256(p)) for k,p in dict(image=image,metadata=metadata[0],**companions).items()}
        from rasterio.warp import transform_bounds
        from encoder_v3.enmap_admission import overlap
        with rasterio.open(image) as ds:
            bounds=transform_bounds(ds.crs,'EPSG:4326',*ds.bounds,densify_pts=21)
            if any(r['role']=='VAL' and r.get('bbox') and overlap(bounds,r['bbox']) for r in existing_members):
                raise ValueError('EnMAP footprint intersects frozen VAL; requires explicit independent role isolation')
            qa={k:checked_raster(p,ds) for k,p in companions.items()}
            try:
                if qa['PIXELMASK'].count!=224:raise ValueError('Per-band defective mask width mismatch')
                for _,window in ds.block_windows(1):
                    data=ds.read(window=window)
                    valid=np.isin(qa['CLASSES'].read(1,window=window),[1,2])
                    for name in ('CLOUD','CLOUDSHADOW','HAZE','CIRRUS','SNOW','TESTFLAGS'):valid &= qa[name].read(1,window=window)==0
                    keep=valid[None]&(qa['PIXELMASK'].read(window=window)==0)&(data!=ds.nodata)&np.isfinite(data)
                    for i,band in enumerate(bands):
                        x=data[i][keep[i]].astype(np.float64)*band['gain']+band['offset']
                        counts[i]+=len(x);sums[i]+=x.sum();squares[i]+=(x*x).sum()
            finally:
                for q in qa.values():q.close()
            rows.append(dict(id='enmap_'+prefix,role='TRAIN',sensor='EnMAP',instrument='HSI',date=__import__('datetime').datetime.strptime(__import__('re').search(r'_(\d{8}T\d{6}Z)_',prefix).group(1),'%Y%m%dT%H%M%SZ').isoformat()+'Z',
                bands=[dict(band_key=b['channel_id'],shape=[ds.height,ds.width],transform=list(ds.transform)[:6],crs=str(ds.crs)) for b in bands],
                shape=[ds.height,ds.width],affine=list(ds.transform)[:6],crs=str(ds.crs),catalog=bands,assets=assets,
                cache_sha256=assets['image']['sha256'],units='surface_reflectance',qa_policy=QA_POLICY,bbox=list(bounds),
                provenance='original native 30m L2A; no resampling or artificial filled bands',lineage_group=prefix))
    stats={}
    for i in range(224):
        name=f'HSI:B{i+1:03d}'
        if counts[i]:
            mean=sums[i]/counts[i];variance=max(0.,squares[i]/counts[i]-mean*mean)
            if variance<=0:raise ValueError('Degenerate valid TRAIN band statistics')
            stats[name]=dict(mean=mean,std=variance**.5,valid_pixels=int(counts[i]),status='VALID')
        else:stats[name]=dict(mean=None,std=None,valid_pixels=0,status='NO_VALID_TRAIN_SUPPORT')
    (output/'statistics.json').write_text(json.dumps(dict(scope='TRAIN-only QA-filtered native EnMAP reflectance',bands=stats),indent=2))
    (output/'members.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    document=dict(status='PASS',qa_policy=QA_POLICY,members_sha256=sha256(output/'members.jsonl'),statistics_sha256=sha256(output/'statistics.json'),
        train_members=len(rows),val_members=0,catalog_total=224,empty_band_ids=[k for k,v in stats.items() if not v['valid_pixels']],
        bad_band_policy='retain original 224 catalog ranks and full geometry; no valid content or targets from invalid bands; no invented mean/std',
        source_spec='https://www.enmap.org/data/doc/EN-PCV-ICD-2009-2_HSI_Product_Specification_Level1_Level2.pdf')
    (output/'release.json').write_text(json.dumps(document,indent=2));return document


def build_window_release(parent,proposal,output):
    if (output/'release.json').exists():raise ValueError('Fresh window release required')
    output.mkdir(parents=True,exist_ok=True)
    parents=[json.loads(line) for line in (parent/'members.jsonl').read_text().splitlines()]
    proposed=json.loads(proposal.read_text())['rows'];members={};rejected=[]
    for item in proposed:
        if not item['inside']:continue
        if item['original_role']!='TRAIN':raise ValueError('Window role must preserve original TRAIN assignment')
        source=next(r for r in parents if r['assets']['image']['path']==str(Path(item['source']).resolve()))
        signature=(item['source'],tuple(item['window']))
        if signature in members:
            members[signature]['aliases'].append(item['id']);continue
        row=json.loads(json.dumps(source));window=rasterio.windows.Window(*item['window'])
        with rasterio.open(source['assets']['image']['path']) as ds:
            transform=ds.window_transform(window);bounds=rasterio.windows.bounds(window,ds.transform)
            from rasterio.warp import transform_bounds
            row.update(id='enmap_native_'+item['id'],window=item['window'],shape=[int(window.height),int(window.width)],
                affine=list(transform)[:6],bbox=list(transform_bounds(ds.crs,'EPSG:4326',*bounds)),aliases=[item['id']])
        for band in row['bands']:band.update(shape=row['shape'],transform=row['affine'])
        members[signature]=row
    sums=np.zeros(224);squares=np.zeros(224);counts=np.zeros(224,dtype=np.int64)
    for row in members.values():
        with rasterio.open(row['assets']['image']['path']) as ds:
            qa={k:checked_raster(v['path'],ds) for k,v in row['assets'].items() if k not in ('image','metadata')}
            w=rasterio.windows.Window(*row['window'])
            try:
                valid=np.isin(qa['CLASSES'].read(1,window=w),[1,2])
                for name in ('CLOUD','CLOUDSHADOW','HAZE','CIRRUS','SNOW','TESTFLAGS'):valid &= qa[name].read(1,window=w)==0
                data=ds.read(window=w);keep=valid[None]&(qa['PIXELMASK'].read(window=w)==0)&(data!=ds.nodata)&np.isfinite(data)
                row['valid_pixels_per_band']=keep.sum((1,2)).tolist()
                if not keep.any():raise ValueError('All-invalid proposed window requires explicit rejection record')
                for i,band in enumerate(row['catalog']):
                    x=data[i][keep[i]].astype(np.float64)*band['gain']+band['offset']
                    counts[i]+=len(x);sums[i]+=x.sum();squares[i]+=(x*x).sum()
            finally:
                for q in qa.values():q.close()
    stats={}
    for i in range(224):
        key=f'HSI:B{i+1:03d}'
        if not counts[i]:stats[key]=dict(mean=None,std=None,valid_pixels=0,status='NO_VALID_TRAIN_SUPPORT')
        else:
            mean=sums[i]/counts[i];variance=squares[i]/counts[i]-mean*mean
            if variance<=0:raise ValueError('Degenerate QA-filtered training variance')
            stats[key]=dict(mean=mean,std=variance**.5,valid_pixels=int(counts[i]),status='VALID')
    (output/'statistics.json').write_text(json.dumps(dict(scope='TRAIN-only QA-filtered reconstructed native EnMAP windows',bands=stats),indent=2))
    (output/'members.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in members.values()))
    doc=dict(status='PASS',context='user-confirmed complete original-window footprint on native lattice; no interpolation',
        qa_policy=QA_POLICY,members_sha256=sha256(output/'members.jsonl'),statistics_sha256=sha256(output/'statistics.json'),
        parent_release_sha256=sha256(parent/'release.json'),proposal_sha256=sha256(proposal),train_members=len(members),
        original_window_records=sum(len(r['aliases']) for r in members.values()),independent_scenes=len({r['lineage_group'] for r in members.values()}),val_members=0,catalog_total=224,
        empty_band_ids=[k for k,v in stats.items() if not v['valid_pixels']])
    (output/'release.json').write_text(json.dumps(doc,indent=2));return doc


class NativeEnmapDataset:
    def __init__(self,path,catalogs):
        self.root=Path(path);doc=json.loads((self.root/'release.json').read_text())
        if doc['status']!='PASS':raise ValueError('EnMAP not admitted')
        checked_file(self.root/'members.jsonl',doc['members_sha256']);checked_file(self.root/'statistics.json',doc['statistics_sha256'])
        self.rows=[json.loads(line) for line in (self.root/'members.jsonl').read_text().splitlines()]
        self.statistics=json.loads((self.root/'statistics.json').read_text());self.catalogs=catalogs
        self.identities=dict(members=doc['members_sha256'],statistics=doc['statistics_sha256'],qa_policy=doc['qa_policy'])
        self.verified={}
        self.input_cache=OrderedDict();self.input_cache_bytes=0;self.input_cache_limit=256*1024**2;self.input_cache_hits=0

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        for asset in row['assets'].values():checked_file(asset['path'],asset['sha256'],self.verified)
        if row['id'] in self.input_cache:
            self.input_cache_hits+=1;self.input_cache.move_to_end(row['id']);return self._copy_observation(self.input_cache[row['id']])
        bands=[]
        with rasterio.open(row['assets']['image']['path']) as ds:
            qa={k:checked_raster(v['path'],ds) for k,v in row['assets'].items() if k not in ('image','metadata')}
            try:
                window=rasterio.windows.Window(*row['window']) if row.get('window') else None
                spatial=np.isin(qa['CLASSES'].read(1,window=window),[1,2])
                for name in ('CLOUD','CLOUDSHADOW','HAZE','CIRRUS','SNOW','TESTFLAGS'):spatial &= qa[name].read(1,window=window)==0
                cube=ds.read(window=window);defects=qa['PIXELMASK'].read(window=window)
                for i,meta in enumerate(row['catalog']):
                    raw=cube[i];valid=spatial&(raw!=ds.nodata)&np.isfinite(raw)&(defects[i]==0)
                    values=np.where(valid,raw*meta['gain']+meta['offset'],0).astype(np.float32)
                    # SRF is absent and unused by LR; empty tensors do not invent a measured response.
                    bands.append(BandObservation(torch.from_numpy(values),torch.from_numpy(valid),torch.empty(0),torch.empty(0),tuple(row['affine']),row['crs'],meta['channel_id']))
            finally:
                for q in qa.values():q.close()
        observation=Observation(bands,row)
        size=sum(b.reflectance.numel()*b.reflectance.element_size()+b.valid_mask.numel()*b.valid_mask.element_size() for b in bands)
        if size<=self.input_cache_limit:
            while self.input_cache and self.input_cache_bytes+size>self.input_cache_limit:
                _,old=self.input_cache.popitem(last=False)
                self.input_cache_bytes-=sum(b.reflectance.numel()*b.reflectance.element_size()+b.valid_mask.numel()*b.valid_mask.element_size() for b in old.bands)
            self.input_cache[row['id']]=observation;self.input_cache_bytes+=size
            return self._copy_observation(observation)
        return observation

    @staticmethod
    def _copy_observation(observation):
        bands=[BandObservation(b.reflectance.clone(),b.valid_mask.clone(),b.wavelengths_nm,b.response,b.affine,b.crs,b.record_id,
               None if b.visible is None else b.visible.clone()) for b in observation.bands]
        return Observation(bands,copy.deepcopy(observation.metadata))

    channels=LRDataset.channels


class CombinedDataset:
    def __init__(self,base,enmap):
        self.datasets=(base,enmap);self.rows=base.rows+enmap.rows;self.offset=len(base.rows)
        self.identities=dict(base=base.identities,enmap=enmap.identities)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return self.datasets[index>=self.offset][index-self.offset if index>=self.offset else index]
    def channels(self,observation,device):return self.datasets[observation.metadata['sensor']=='EnMAP'].channels(observation,device)
