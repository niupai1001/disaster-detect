"""USE wrapper keeps labels separate from source-derived observation quality."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from .segformer_dataset import S2SegDataset, S2_10_BANDS, _augment
from .optical_observation import OpticalObservation, SUBSETS, training_subset
from ..models.spectral_contract import wavelengths_for
from ..config import PROJECT_ROOT


def augmentation_grid_map(height, width, seed, enabled):
    """Map augmented pixel corners back to the original image grid."""
    mapping=np.eye(3)
    flags=dict(flip_horizontal=False,flip_vertical=False,quarter_turns_ccw=0)
    if not enabled:return mapping,flags
    rng=np.random.default_rng(seed)
    flags['flip_horizontal']=bool(rng.random()<.5)
    flags['flip_vertical']=bool(rng.random()<.5)
    k=int(rng.integers(0,4));flags['quarter_turns_ccw']=k
    if flags['flip_horizontal']:mapping=mapping@np.array([[-1,0,width],[0,1,0],[0,0,1]])
    if flags['flip_vertical']:mapping=mapping@np.array([[1,0,0],[0,-1,height],[0,0,1]])
    rotations=[np.eye(3),np.array([[0,-1,width],[1,0,0],[0,0,1]]),
               np.array([[-1,0,width],[0,-1,height],[0,0,1]]),
               np.array([[0,1,0],[-1,0,height],[0,0,1]])]
    return mapping@rotations[k],flags


class USESegDataset(S2SegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixed, self.eval_subset, self.exposure = False, 'full', 0
        self.quality_root = PROJECT_ROOT / 'outputs/use/quality'
        self.quality = json.loads((self.quality_root / 'index.json').read_text(encoding='utf-8'))

    def __getitem__(self, index):
        rec = self.records[index]
        # Derive augmentation twice with the same seed, never use the label mask as input.
        sample = super().__getitem__(index)
        entry = self.quality[rec['sample_id']]
        if entry['cache_sha256'] != rec['cache_sha256'] or entry['time_index'] != rec['time_index']:
            raise ValueError('USE quality provenance differs from selected observation')
        quality_file = self.quality_root / entry['file']
        if hashlib.sha256(quality_file.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('USE quality sidecar hash changed')
        with np.load(quality_file) as data:
            quality = data['pixel_valid'].copy()
        source_height,source_width=quality.shape[1:]
        augmentation_seed=(self.seed * 1_000_003 + self.epoch * 7_919 + index) % (2**63 - 1)
        grid_map,augmentation=augmentation_grid_map(source_height,source_width,augmentation_seed,self.augment)
        if self.augment:
            rng = np.random.default_rng(augmentation_seed)
            quality, _, _ = _augment(quality, np.zeros(quality.shape[1:]), np.zeros(quality.shape[1:]), rng)
        metadata = torch.tensor([[wave, 0., 10. if band in ('B02','B03','B04','B08') else 20.]
                                 for band, wave in zip(S2_10_BANDS, wavelengths_for(S2_10_BANDS))])
        known = torch.ones(10, 3, dtype=torch.bool)
        known[:, 1] = False
        xmin,ymin,xmax,ymax = rec['bbox']
        original_grid=np.array([[(xmax-xmin)/source_width,0,xmin],[0,-(ymax-ymin)/source_height,ymax],[0,0,1]])
        observation = OpticalObservation(sample['image'], S2_10_BANDS, metadata, known,
            torch.ones(10, dtype=torch.bool), torch.from_numpy(quality),
            {'crs': rec['crs'], 'bbox': rec['bbox'], 'units': 'core25_zscore',
             'sensor': 'Sentinel-2', 'product_level': 'L2A', 'processing': 'S2_harmonized', 'date': rec['date'], 'target_gsd_m': 10,
             'band_metadata_source': 'spectral_contract.S2_WAVELENGTH_UM nominal; per-product platform response unknown',
             'physical_scale': .0001, 'physical_offset': 0.,
             'source_transform': original_grid[:2].flatten().tolist(),
             'transform': (original_grid@grid_map)[:2].flatten().tolist(),'augmentation':augmentation})
        subset = training_subset(self.seed, rec['sample_id'], self.exposure) if self.mixed else self.eval_subset
        sample['image'] = observation.subset(SUBSETS[subset])
        return sample
