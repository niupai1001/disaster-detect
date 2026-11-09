"""Read the new optical data release without implicit band padding or resampling."""
from pathlib import Path
import json
import hashlib
import numpy as np


class ExpandedOpticalDataset:
    def __init__(self, release, role='TRAIN'):
        self.root=Path(release)
        if role not in ['TRAIN','VAL']:raise ValueError('Only TRAIN/VAL input access is released')
        meta=json.loads((self.root/'release.json').read_text(encoding='utf-8'))
        if not meta['data_release_accepted']:raise ValueError('Data release not finalized')
        path=self.root/'members.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['manifest_sha256']:raise ValueError('Frozen data members changed')
        rows=[json.loads(t) for t in path.read_text(encoding='utf-8').splitlines()]
        self.rows=[r for r in rows if r['role']==role]

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        r=self.rows[index];bands=[]
        with np.load(r['cache'],allow_pickle=False) as z:
            for b in r['bands']:
                k=b['band_key'];q=z[k+'__valid'].copy()
                bands.append(dict(b,reflectance=z[k].copy(),pixel_valid=q,units='surface_reflectance',values_already_scaled=True))
        return dict(bands=bands,product_id=r['product'],instrument_id=r['instrument'],timestamp=r['date'],
                    processing_domain=r['processing_domain'],role=r['role'],id=r['id'],
                    srf_status=r['srf_scientific_ingestion'],training_authorized=False)


class G1OpticalDataset(ExpandedOpticalDataset):
    """Join accepted physical SRF metadata without changing frozen pixel values."""
    def __init__(self, release, gate, role='TRAIN'):
        super().__init__(release,role)
        gate=Path(gate)
        checks=json.loads((gate/'data_contract_tests.json').read_text(encoding='utf-8'))
        if checks['status']!='PASS':raise ValueError('G1 physical acceptance failed')
        path=gate/'observation_contracts.jsonl'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checks['observation_contracts_sha256']:
            raise ValueError('G1 observation metadata changed')
        self.contracts={c['id']:c for c in map(json.loads,path.read_text(encoding='utf-8').splitlines())}
        srf=gate/'srf_manifest.json'
        if hashlib.sha256(srf.read_bytes()).hexdigest()!=checks['srf_manifest_sha256']:
            raise ValueError('G1 spectral metadata changed')
        self.curves={c['id']:c for c in json.loads(srf.read_text(encoding='utf-8'))['curves']}
        if any(r['id'] not in self.contracts for r in self.rows):raise ValueError('Missing G1 observation contract')

    def __getitem__(self,index):
        item=super().__getitem__(index);contract=self.contracts[item['id']]
        bands={b['band_key']:b for b in contract['bands']}
        for b in item['bands']:
            meta=bands[b['band_key']];b['spectral_response_ref']=dict(self.curves[meta['spectral_response_ref']])
            b['wavelength_nm']=meta['center_nm'];b['fwhm_nm']=meta['fwhm_nm']
            a,d,e,f,g,h=b['transform'];height,width=b['shape']
            b['footprint']=dict(crs=b['crs'],corners=[(a*x+d*y+e,f*x+g*y+h) for x,y in [(0,0),(width,0),(width,height),(0,height)]])
        item['processing_baseline']=contract['processing_baseline']
        item['srf_status']='G1_verified_instrument_specific'
        return item
