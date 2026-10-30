"""Independent-grid v2 input/output contracts. No formal training entry point."""
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EOAsset:
    modality: str
    channel_names: tuple
    image: torch.Tensor
    channel_metadata: torch.Tensor
    metadata_known: torch.Tensor
    sample_grid: dict
    native_resolution: Optional[tuple]
    effective_resolution: Optional[tuple]
    pixel_valid: torch.Tensor
    acquisition_time: str
    radiometry: str
    provenance: dict

    def validate(self):
        if self.modality not in ('s2', 'landsat', 's1', 'hr'):
            raise ValueError('Unknown modality')
        if self.image.ndim != 3 or self.image.shape != self.pixel_valid.shape:
            raise ValueError('Expected channel-first image and per-channel valid mask')
        if self.pixel_valid.dtype != torch.bool:
            raise ValueError('Pixel validity must be boolean')
        if len(self.channel_names) != self.image.shape[0] or not self.pixel_valid.any():
            raise ValueError('Empty observation or channel mismatch')
        if not torch.isfinite(self.image[self.pixel_valid]).all():
            raise ValueError('Nonfinite valid input')
        if not self.sample_grid.get('crs') or len(self.sample_grid.get('transform', ())) != 6:
            raise ValueError('Asset requires its own georeferenced grid')
        a,b,c,d,e,f=self.sample_grid['transform']
        if not all(torch.isfinite(torch.tensor(v)) for v in (a,b,c,d,e,f)) or abs(a*e-b*d)<1e-10:
            raise ValueError('Invalid affine grid')
        if self.modality in ('s2','landsat'):
            if self.channel_metadata.shape!=(len(self.image),3) or self.metadata_known.shape!=self.channel_metadata.shape:
                raise ValueError('Optical channel metadata shape mismatch')
            if self.metadata_known.dtype!=torch.bool:
                raise ValueError('Metadata known mask must be boolean')
            known_values=torch.where(self.metadata_known,self.channel_metadata,0)
            if not torch.isfinite(known_values).all():raise ValueError('Nonfinite known metadata')
            waves=self.channel_metadata[:,0][self.metadata_known[:,0]]
            if ((waves<.1)|(waves>3)).any():raise ValueError('Invalid optical wavelengths')
        return self


@dataclass
class EOObservation:
    site_id: str
    footprint: object
    assets: tuple[EOAsset, ...]
    role: str
    relation: dict

    def validate(self):
        if not self.assets or self.role not in ('TRAIN', 'PRETRAIN_VAL'):
            raise ValueError('Empty or disallowed observation')
        for asset in self.assets:
            asset.validate()
        return self


@dataclass
class FeatureLevel:
    tensor: torch.Tensor
    valid_mask: torch.Tensor
    transform: tuple
    grid_spacing_m: float
    receptive_field_m: Optional[float]
    origin_modality: str
    source_native_resolution: Optional[tuple]
    interpolated: bool
    crs: Optional[str] = None


@dataclass
class USEOutput:
    native_detail: Optional[FeatureLevel] = None
    metric_local: Optional[FeatureLevel] = None
    metric_patch: Optional[FeatureLevel] = None
    optical_semantic: Optional[FeatureLevel] = None
    sar_semantic: Optional[FeatureLevel] = None
    shared_metric_tokens: Optional[FeatureLevel] = None
    global_embedding: Optional[torch.Tensor] = None
    semantic_levels: Optional[dict] = None


def load_asset(dataset, index):
    """Reuse the accepted loader's hash check while retaining the entire source row."""
    row = dataset.rows[index]
    item = dataset[index]
    if row['sensor'] in ('s2', 'landsat'):
        image, valid = item.x, item.pixel_valid
        metadata, known = item.band_metadata, item.known_mask
    else:
        image, valid = item['x'], item['pixel_valid']
        metadata = torch.empty((len(image), 0))
        known = torch.empty((len(image), 0), dtype=torch.bool)
    return EOAsset(row['sensor'], tuple(row['band_ids']), image, metadata, known,
                   {k: row.get(k) for k in ('crs', 'transform', 'bbox', 'grid_gsd_m')},
                   row.get('native_resolution'), row.get('effective_resolution'), valid,
                   row['date'], row['units'], dict(row)).validate()
