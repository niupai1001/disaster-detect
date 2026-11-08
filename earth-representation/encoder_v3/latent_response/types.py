"""Static sensor/channel identity and native-grid contract; no learned state."""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class SensorKey:
    platform: str
    instrument: str

    def __post_init__(self):
        for name in ('platform', 'instrument'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or '|' in value:
                raise ValueError(f'SensorKey.{name} must be a nonempty string without "|"')

    @property
    def uid(self):
        return f'{self.platform}|{self.instrument}'


@dataclass(frozen=True)
class ChannelMeta:
    """Stable identity inside the native channel catalog, never the tensor index."""
    channel_id: str
    canonical_ordinal: int
    catalog_total_channels: int

    def __post_init__(self):
        if not isinstance(self.channel_id, str) or not self.channel_id:
            raise ValueError('channel_id required')
        if type(self.canonical_ordinal) is not int or type(self.catalog_total_channels) is not int:
            raise ValueError('integer canonical ordinal and catalog size required')
        if not 0 <= self.canonical_ordinal < self.catalog_total_channels:
            raise ValueError('canonical ordinal must lie inside the native catalog')


def metric_projected_crs(value):
    """Parse a CRS and require projected metre units; never reproject silently."""
    if not isinstance(value, str) or not value:
        raise ValueError('Explicit CRS required')
    try:
        from rasterio.crs import CRS
        crs = CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f'Unknown or unparsable CRS {value!r}; refusing a metric interpretation') from exc
    if not crs.is_projected or crs.linear_units not in ('metre', 'meter'):
        raise ValueError('Metric projected CRS required; geographic, imperial and unknown CRS are rejected')
    return value


@dataclass(frozen=True)
class NativeGrid:
    """North-up projected metric grid; rotated/geographic grids stay unsupported."""
    affine: tuple
    shape: tuple
    crs: str

    def __post_init__(self):
        if len(self.affine) != 6 or len(self.shape) != 2:
            raise ValueError('six affine terms and two shape entries required')
        a, b, _, d, e, _ = self.affine
        if not all(math.isfinite(float(v)) for v in self.affine) or b or d or a <= 0 or e >= 0:
            raise ValueError('Reference path requires north-up projected metric grids')
        if any(type(v) is not int or v < 1 for v in self.shape):
            raise ValueError('Positive integer native shape required')
        metric_projected_crs(self.crs)

    @property
    def gsd_xy(self):
        return float(self.affine[0]), float(-self.affine[4])


@dataclass
class ChannelObservation:
    """One native channel after the confirmed input transform; values are not re-scaled here."""
    sensor_key: SensorKey
    channel_meta: ChannelMeta
    values: torch.Tensor
    valid: torch.Tensor
    visible: torch.Tensor
    affine: tuple
    crs: str

    def __post_init__(self):
        self.validate()

    def validate(self):
        if not isinstance(self.sensor_key, SensorKey) or not isinstance(self.channel_meta, ChannelMeta):
            raise ValueError('Typed sensor key and channel metadata required')
        if self.values.ndim != 2 or not self.values.is_floating_point():
            raise ValueError('Floating [H,W] native values required')
        for name, mask in (('valid', self.valid), ('visible', self.visible)):
            if mask.shape != self.values.shape or mask.dtype is not torch.bool:
                raise ValueError(f'Boolean {name} mask matching values required')
        NativeGrid(self.affine, tuple(self.values.shape), self.crs)
        keep = self.keep()
        if bool(keep.any()) and not bool(torch.isfinite(self.values[keep]).all()):
            raise ValueError('Nonfinite visible scalar')
        return self

    def keep(self):
        return self.valid & self.visible

    @property
    def grid(self):
        return NativeGrid(self.affine, tuple(self.values.shape), self.crs)

    @property
    def gsd_xy(self):
        return self.grid.gsd_xy
