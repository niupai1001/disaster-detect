"""Resolve physical channel identity from product metadata or a traceable catalog."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import re


class MissingMetadata(ValueError):
    pass


QUANTITIES = ('surface_reflectance', 'radiance', 'DN', 'gamma0_linear', 'sigma0_linear', 'gamma0_db', 'sigma0_db')
POLARIZATIONS = ('none', 'VV', 'VH', 'HH', 'HV')


@dataclass(frozen=True)
class PhysicalChannel:
    modality: str
    quantity: str
    gx: float
    gy: float
    center_um: float | None = None
    fwhm_um: float | None = None
    frequency_ghz: float | None = None
    polarization: str = 'none'
    incidence_deg: float | None = None
    source: str = ''

    def __post_init__(self):
        if self.modality not in ('optical', 'sar') or self.quantity not in QUANTITIES:
            raise MissingMetadata('Explicit supported modality and physical quantity required')
        for name in ('gx', 'gy', 'center_um', 'fwhm_um', 'frequency_ghz'):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise MissingMetadata('Invalid physical field: '+name)
        if self.incidence_deg is not None and not (math.isfinite(self.incidence_deg) and 0 <= self.incidence_deg <= 90):
            raise MissingMetadata('Incidence must be in [0,90] degrees')
        if self.modality == 'optical':
            if self.center_um is None or self.quantity not in QUANTITIES[:3] or self.polarization != 'none':
                raise MissingMetadata('Optical channel requires wavelength and an optical quantity')
            if self.frequency_ghz is not None:
                raise MissingMetadata('Do not mix radar frequency with optical wavelength')
        elif self.frequency_ghz is None or self.polarization not in POLARIZATIONS[1:] or self.quantity not in QUANTITIES[3:]:
            raise MissingMetadata('SAR requires frequency, polarization and calibrated backscatter quantity')
        elif self.center_um is not None or self.fwhm_um is not None:
            raise MissingMetadata('Optical FWHM is not radar bandwidth')
        if not self.source:
            raise MissingMetadata('Physical metadata provenance required')

    def encoded(self):
        # Logs use fixed units: metres, micrometres, GHz; missing is distinct from log(1).
        fields = (self.gx, self.gy, self.center_um, self.fwhm_um, self.frequency_ghz)
        values = [0. if v is None else math.log(v) for v in fields]
        values.append(0. if self.incidence_deg is None else self.incidence_deg/90.)
        flags = [float(v is not None) for v in (*fields, self.incidence_deg)]
        categories = [int(self.modality == 'sar'), QUANTITIES.index(self.quantity), POLARIZATIONS.index(self.polarization)]
        return values+flags, categories

    def as_dict(self):
        return asdict(self)


class MetadataResolver:
    def __init__(self, catalog_path):
        path = Path(catalog_path)
        self.catalog = json.loads(path.read_text(encoding='utf-8-sig'))
        self.catalog_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.catalog_source = str(path.resolve())+'#sha256='+self.catalog_sha

    def resolve(self, row, band):
        name = band['band_key']
        transform = band['transform']
        gx, gy = float(transform[0]), float(-transform[4])
        sensor = row.get('sensor', '')
        modality = row.get('modality', 'optical' if sensor in ('S2', 'L8', 'L9', 'EnMAP') else None)
        quantity = row.get('units')
        # Explicit sidecar/product fields take precedence; their units are part of the API.
        external = row.get('channel_metadata', {}).get(name)
        if external is not None:
            allowed = {'center_um', 'fwhm_um', 'frequency_ghz', 'polarization', 'incidence_deg', 'source'}
            if set(external)-allowed:
                raise MissingMetadata('Unknown or unit-ambiguous channel metadata fields')
            return PhysicalChannel(modality, quantity, gx, gy, **external)
        if sensor == 'EnMAP':
            entries = {x['channel_id']: x for x in row.get('catalog', [])}
            if name not in entries:
                raise MissingMetadata('EnMAP requires the matching product XML channel catalog')
            item = entries[name]
            asset = row.get('assets', {}).get('metadata', {})
            source = str(asset.get('path', ''))+'#sha256='+str(asset.get('sha256', ''))
            if not asset.get('path') or not asset.get('sha256'):
                raise MissingMetadata('EnMAP XML provenance missing')
            return PhysicalChannel(modality, quantity, gx, gy, item['center_nm']/1000., item['fwhm_nm']/1000., source=source)
        key = {'L8': 'LANDSAT8_OLI', 'L9': 'LANDSAT9_OLI2'}.get(sensor, row.get('instrument', sensor))
        records = self.catalog.get('sensors', {}).get(key)
        if modality != 'optical' or records is None:
            raise MissingMetadata(f'{sensor}/{name}: product channel_metadata or a traceable catalog is required')
        canonical = re.sub(r'^SR_', '', name)
        canonical = re.sub(r'^B0+(\d)', r'B\1', canonical)
        matches = [r for r in records['bands'] if r['band'] == canonical]
        if len(matches) != 1:
            raise MissingMetadata(f'{key}/{name}: unique catalog wavelength missing')
        item = matches[0]
        return PhysicalChannel(modality, quantity, gx, gy, item['centre_nm']/1000.,
                               None if item.get('fwhm_nm') is None else item['fwhm_nm']/1000., source=self.catalog_source+';'+key+'/'+canonical)

    def audit(self, rows):
        resolved, blocked = [], []
        for row in rows:
            for band in row['bands']:
                try:
                    value = self.resolve(row, band)
                    resolved.append(dict(id=row['id'], role=row['role'], channel=band['band_key'], physical=value.as_dict()))
                except MissingMetadata as exc:
                    blocked.append(dict(id=row['id'], role=row['role'], channel=band['band_key'], reason=str(exc)))
        return dict(status='PASS' if not blocked else 'BLOCKED', catalog_sha256=self.catalog_sha,
                    observations=len(rows), resolved_channels=len(resolved), blocked=blocked, channels=resolved)
