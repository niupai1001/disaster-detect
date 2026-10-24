"""Variable-band optical inputs; supervision masks deliberately stay outside this API."""
from dataclasses import dataclass, replace
import hashlib

import torch
from torch import Tensor


SUBSETS = {
    'rgb': ('B02', 'B03', 'B04'),
    'rgb_nir': ('B02', 'B03', 'B04', 'B08'),
    'six': ('B02', 'B03', 'B04', 'B08', 'B11', 'B12'),
    'unseen5': ('B02', 'B03', 'B04', 'B8A', 'B11'),
    'full': ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'),
}


class OpticalBatch(dict):
    def to(self, device, **kwargs):
        return OpticalBatch({k: v.to(device, **kwargs) if isinstance(v, Tensor) else v for k, v in self.items()})

    def size(self, dim):
        return self['x'].size(dim)


def training_subset(seed: int, sample_id: str, exposure: int) -> str:
    """Stateless 1/6 RGB, 1/6 RGB+NIR, 1/6 six, 1/2 full sampling."""
    if exposure < 0:
        raise ValueError('Exposure must be nonnegative')
    digest = hashlib.sha256(f'{seed}\0{sample_id}\0{exposure}'.encode()).digest()
    value = int.from_bytes(digest[:8], 'big') / 2**64
    return ('rgb', 'rgb_nir', 'six')[int(value * 6)] if value < .5 else 'full'


@dataclass(frozen=True)
class OpticalObservation:
    """Unbatched C,H,W reflectance/features with explicit physical metadata.

    Metadata columns: wavelength_um, fwhm_um, native_gsd_m. Unknown entries
    must be zero with known_mask=False. Pixel validity must be derived from
    imagery, never from label availability. Spatial metadata is provenance,
    not a learned feature; include product, units, processing, CRS and grid.
    """
    x: Tensor
    band_ids: tuple[str, ...]
    band_metadata: Tensor
    known_mask: Tensor
    band_valid: Tensor
    pixel_valid: Tensor
    spatial_metadata: dict

    def validate(self):
        if self.x.ndim != 3 or min(self.x.shape) < 1 or not self.x.is_floating_point():
            raise ValueError('Expected nonempty floating C,H,W')
        c, h, w = self.x.shape
        if len(self.band_ids) != c or len(set(self.band_ids)) != c:
            raise ValueError('Band identities must be unique and match C')
        if self.band_metadata.shape != (c, 3) or self.known_mask.shape != (c, 3):
            raise ValueError('Expected C,3 physical metadata and known mask')
        if self.band_valid.shape != (c,) or self.pixel_valid.shape != (c, h, w):
            raise ValueError('Validity shapes differ from observation')
        for mask in (self.known_mask, self.band_valid, self.pixel_valid):
            if mask.dtype != torch.bool or mask.device != self.x.device:
                raise ValueError('Masks must be boolean on the image device')
        if self.band_metadata.device != self.x.device or not self.band_metadata.is_floating_point():
            raise ValueError('Metadata must be floating on the image device')
        if not torch.isfinite(self.band_metadata).all():
            raise ValueError('Nonfinite metadata')
        if (self.band_metadata[~self.known_mask] != 0).any():
            raise ValueError('Unknown metadata must use explicit zero placeholders')
        if (self.band_metadata[self.known_mask] <= 0).any():
            raise ValueError('Known wavelengths, bandwidths and GSD must be positive')
        wave = self.band_metadata[:, 0][self.known_mask[:, 0]]
        if ((wave < .1) | (wave > 3)).any():
            raise ValueError('V1 optical wavelengths must be in micrometres, within .1 to 3')
        valid = self.pixel_valid & self.band_valid[:, None, None]
        if not valid.any() or not torch.isfinite(self.x[valid]).all():
            raise ValueError('No valid observation or nonfinite valid pixels')
        return self

    def subset(self, bands):
        self.validate()
        bands = tuple(bands)
        if not bands or len(set(bands)) != len(bands):
            raise ValueError('Subset must contain unique bands')
        indices = [self.band_ids.index(b) for b in bands]
        return replace(self, x=self.x[indices], band_ids=bands,
                       band_metadata=self.band_metadata[indices], known_mask=self.known_mask[indices],
                       band_valid=self.band_valid[indices], pixel_valid=self.pixel_valid[indices]).validate()


def collate_optical(observations):
    """Pad only the channel axis, masking padding before downstream computation."""
    if not observations:
        raise ValueError('Empty batch')
    for obs in observations:
        obs.validate()
    first = observations[0]
    if any(o.x.shape[1:] != first.x.shape[1:] or o.x.device != first.x.device
           or o.x.dtype != first.x.dtype for o in observations):
        raise ValueError('Batch requires a common grid shape, device and dtype')
    b, c = len(observations), max(o.x.shape[0] for o in observations)
    h, w = first.x.shape[1:]
    x = first.x.new_zeros(b, c, h, w)
    metadata = first.band_metadata.new_zeros(b, c, 3)
    known = torch.zeros(b, c, 3, dtype=torch.bool, device=x.device)
    bands = torch.zeros(b, c, dtype=torch.bool, device=x.device)
    pixels = torch.zeros(b, c, h, w, dtype=torch.bool, device=x.device)
    for i, obs in enumerate(observations):
        n = obs.x.shape[0]
        valid = obs.pixel_valid & obs.band_valid[:, None, None]
        x[i, :n] = torch.where(valid, obs.x, 0)
        metadata[i, :n], known[i, :n] = obs.band_metadata, obs.known_mask
        bands[i, :n], pixels[i, :n] = obs.band_valid, valid
    return OpticalBatch(x=x, band_metadata=metadata, known_mask=known, band_valid=bands,
                pixel_valid=pixels, band_ids=[o.band_ids for o in observations],
                spatial_metadata=[o.spatial_metadata for o in observations])


def physical_reflectance(dn, scale, offset, image_valid):
    """Apply the declared product transform once; never infer it from sensor name."""
    if dn.shape != image_valid.shape or image_valid.dtype != torch.bool:
        raise ValueError('Image quality mask shape/type mismatch')
    scale = torch.as_tensor(scale, dtype=torch.float32, device=dn.device)
    offset = torch.as_tensor(offset, dtype=torch.float32, device=dn.device)
    if scale.numel() != 1 or offset.numel() != 1 or not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or scale <= 0:
        raise ValueError('Finite scalar scale>0 and offset required')
    valid = image_valid & torch.isfinite(dn)
    rho = torch.where(valid, dn, 0).float() * scale + offset
    return torch.where(valid, rho, 0), valid
