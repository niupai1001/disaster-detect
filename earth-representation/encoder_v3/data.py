"""Read-only bridge to hash-bound native optical caches; no data downloads."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .model import BandObservation


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): result.update(block)
    return result.hexdigest()


def checked_file(path, expected, verified=None):
    path=Path(path)
    stat=path.stat(); signature=(stat.st_size,stat.st_mtime_ns)
    key=(str(path.resolve()),expected)
    if verified is not None and verified.get(key)==signature: return
    if not isinstance(expected,str) or len(expected)!=64 or sha256(path)!=expected:
        raise ValueError(f'File identity mismatch: {path}')
    after=path.stat()
    if signature!=(after.st_size,after.st_mtime_ns): raise ValueError(f'File changed during verification: {path}')
    if verified is not None: verified[key]=signature


@dataclass
class Observation:
    bands: list
    metadata: dict

    @property
    def identifier(self): return self.metadata['id']

    def pin_memory(self):
        bands=[BandObservation(b.reflectance.pin_memory(),b.valid_mask.pin_memory(),b.wavelengths_nm.pin_memory(),b.response.pin_memory(),b.affine,b.crs,b.record_id,None if b.visible is None else b.visible.pin_memory()) for b in self.bands]
        return Observation(bands,self.metadata)


class NativeOpticalDataset:
    def __init__(self, release, physical_contracts, *, role):
        if role not in ('TRAIN','VAL'): raise ValueError('Only explicit TRAIN or VAL access is enabled')
        self.root=Path(release); self.contract_root=Path(physical_contracts)
        metadata=json.loads((self.root/'release.json').read_text(encoding='utf-8-sig'))
        checks=json.loads((self.contract_root/'data_contract_tests.json').read_text(encoding='utf-8-sig'))
        if metadata.get('data_release_accepted') is not True or checks.get('status')!='PASS':
            raise ValueError('Source release lacks physical-contract acceptance')
        member_path=self.root/'members.jsonl'; contract_path=self.contract_root/'observation_contracts.jsonl'; srf_path=self.contract_root/'srf_manifest.json'
        checked_file(member_path,metadata['manifest_sha256'])
        if checks.get('data_manifest_sha256')!=metadata['manifest_sha256']: raise ValueError('Contract belongs to another data release')
        checked_file(contract_path,checks['observation_contracts_sha256']); checked_file(srf_path,checks['srf_manifest_sha256'])
        with member_path.open(encoding='utf-8-sig') as stream: self.rows=[json.loads(line) for line in stream if line.strip()]
        self.rows=[row for row in self.rows if row['role']==role]
        if len({r['id'] for r in self.rows})!=len(self.rows): raise ValueError('Duplicate observation identities')
        with contract_path.open(encoding='utf-8-sig') as stream: self.contracts={row['id']:row for row in map(json.loads,stream)}
        self.curves={c['id']:c for c in json.loads(srf_path.read_text(encoding='utf-8-sig'))['curves']}
        self.identities=dict(members=metadata['manifest_sha256'],contracts=checks['observation_contracts_sha256'],srf=checks['srf_manifest_sha256'])
        if 'statistics_sha256' in metadata:self.identities['statistics']=metadata['statistics_sha256']
        self.release_metadata=metadata
        self.role=role
        self.verified_files={}; self.curve_tensors={}

    def __len__(self): return len(self.rows)

    def engineering_train_stds(self, observation):
        path=self.root/'statistics.json';checked_file(path,self.release_metadata['statistics_sha256'],self.verified_files)
        stats=json.loads(path.read_text(encoding='utf-8-sig'))
        if not stats.get('scope','').startswith('TRAIN-only'):raise ValueError('Statistics are not declared TRAIN-only')
        instrument=observation.metadata['instrument']
        return {b.record_id:stats['bands'][instrument+':'+b.record_id]['std'] for b in observation.bands}

    def __getitem__(self,index):
        row=self.rows[index]; contract=self.contracts[row['id']]
        if row.get('units')!='surface_reflectance' or contract.get('units')!='surface_reflectance':
            raise ValueError('Photographic DN and radar are not optical reflectance')
        if contract['role']!=self.role or contract['cache_sha256']!=row['cache_sha256']:
            raise ValueError('Role or cache identity mismatch')
        if contract.get('timestamp')!=row['date'] or not contract.get('values_already_scaled'):
            raise ValueError('Missing acquisition/calibration contract')
        checked_file(row['cache'],row['cache_sha256'],self.verified_files)
        band_contracts={b['band_key']:b for b in contract['bands']}; bands=[]
        with np.load(row['cache'],allow_pickle=False) as arrays:
            for item in row['bands']:
                key=item['band_key']; physical=band_contracts[key]; grid=physical['grid']
                if tuple(grid['shape'])!=tuple(item['shape']) or tuple(grid['affine_transform'])!=tuple(item['transform']) or grid['crs']!=item['crs']:
                    raise ValueError('Native grid differs from physical contract')
                curve=self.curves[physical['spectral_response_ref']]
                if curve.get('wavelength_unit')!='nm' or curve.get('source_type') not in ('measured_band_averaged_srf','transformed_instrument_rsr') or not curve.get('source_url') or not curve.get('file_hash'):
                    raise ValueError('Unverified SRF provenance')
                checked_file(curve['path'],curve['sha256'],self.verified_files)
                curve_key=(curve['id'],curve['sha256'])
                if curve_key not in self.curve_tensors:
                    with np.load(curve['path'],allow_pickle=False) as srf:
                        self.curve_tensors[curve_key]=(torch.from_numpy(srf['spectral_wavelength_grid_nm'].copy()).float(),torch.from_numpy(srf['normalized_response'].copy()).float())
                wavelengths,response=(value.clone() for value in self.curve_tensors[curve_key])
                values=arrays[key].copy(); mask=arrays[key+'__valid'].copy()
                if values.shape!=tuple(grid['shape']) or mask.dtype!=np.bool_:
                    raise ValueError('Invalid native array shape or mask dtype')
                if not np.isfinite(values[mask]).all(): raise ValueError('Nonfinite valid reflectance')
                band=BandObservation(torch.from_numpy(values).float(),torch.from_numpy(mask),wavelengths,response,tuple(item['transform']),item['crs'],key)
                band.validate(); bands.append(band)
        return Observation(bands,dict(row,physical_contract=contract,v3_admission='engineering_only_not_formal',frozen_source_ids=self.identities))


def to_device(observation, device, *, non_blocking=False):
    bands=[BandObservation(b.reflectance.to(device,non_blocking=non_blocking),b.valid_mask.to(device,non_blocking=non_blocking),b.wavelengths_nm.to(device,non_blocking=non_blocking),b.response.to(device,non_blocking=non_blocking),b.affine,b.crs,b.record_id,None if b.visible is None else b.visible.to(device,non_blocking=non_blocking)) for b in observation.bands]
    return Observation(bands,observation.metadata)
