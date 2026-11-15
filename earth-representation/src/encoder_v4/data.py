"""Frozen optical members, physical metadata, and value-isolated task views."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re

import numpy as np
import torch

from encoder_v3.data import checked_file, sha256
from encoder_v3.shared_mae.metadata import MetadataResolver
from encoder_v3.tasks import task_rng

from .geometry import Grid, WorkGrid, build_work_grid


@dataclass(frozen=True)
class Attribute:
    status: str = 'missing'
    unit: str = ''
    source: str = ''
    summary: tuple[float, ...] = ()
    samples: tuple[tuple[float, float, float, float], ...] = ()  # x, y/direction, response, weight

    def __post_init__(self):
        if self.status not in ('missing', 'observed', 'estimated', 'not_applicable'):
            raise ValueError(f'Unexpected attribute state: {self.status}')
        if self.status in ('observed', 'estimated'):
            if not self.source or not self.unit or not (self.summary or self.samples):
                raise ValueError('Available physical attributes require values, units and provenance')
        elif self.summary or self.samples:
            raise ValueError('Unavailable physical attributes cannot carry observations')
        if (not np.isfinite(self.summary).all() or any(len(p) != 4 for p in self.samples)
                or not np.isfinite(self.samples).all() or any(p[3] <= 0 for p in self.samples)):
            raise ValueError('Nonfinite or malformed physical samples')


@dataclass(frozen=True)
class Physics:
    center: Attribute = Attribute()
    bandwidth: Attribute = Attribute()
    srf: Attribute = Attribute()
    gsd: Attribute = Attribute()
    psf: Attribute = Attribute()
    mtf: Attribute = Attribute()


@dataclass(frozen=True)
class Band:
    key: str
    values: torch.Tensor
    valid: torch.Tensor
    grid: Grid
    physics: Physics


@dataclass(frozen=True)
class Observation:
    id: str
    sensor: str
    bands: tuple[Band, ...]


@dataclass(frozen=True)
class VisibleBand:
    values: torch.Tensor
    observed: torch.Tensor
    grid: Grid
    physics: Physics


@dataclass(frozen=True)
class VisibleObservation:
    bands: tuple[VisibleBand, ...]
    work: WorkGrid


@dataclass(frozen=True)
class TargetRequest:
    physics: Physics
    grid: Grid
    pixels: torch.Tensor
    cells: torch.Tensor
    output_index: torch.Tensor
    area: torch.Tensor


@dataclass(frozen=True)
class TaskView:
    visible: VisibleObservation
    requests: tuple[TargetRequest, ...]
    truth: tuple[torch.Tensor, ...]
    keys: tuple[str, ...]


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def scene(row):
    key = row['acquisition_key']
    if plain := re.fullmatch(r'(S2[A-Z]|LC0[89]):(\d{8}):([0-9A-Z]+)', key):
        return plain.groups()
    if s2 := re.match(r'(S2[A-Z])_MSIL2A_(\d{8})T\d{6}_R\d+_T(\d{2}[A-Z]{3})_', key):
        return s2.groups()
    if landsat := re.match(r'(LC0[89])_L2\w\w_(\d{6})_(\d{8})_', key):
        return landsat[1], landsat[3], landsat[2]
    raise ValueError(f'Unknown acquisition key for {row["id"]}: {key}')


def derive_members(release):
    source = json.loads((release / 'release.json').read_text(encoding='utf-8'))
    if sha256(release / 'members.jsonl') != source['manifest_sha256']:
        raise ValueError('Research release manifest hash mismatch')
    rows = read_rows(release / 'members.jsonl')
    roles = {role: [row for row in rows if row['dataset_role'] == role] for role in ('TRAIN', 'VAL', 'TEST')}
    if len(rows) != 1800 or sum(map(len, roles.values())) != len(rows):
        raise ValueError('Unexpected research release member count or role')
    test_scenes = {scene(row) for row in roles['TEST']}
    excluded = [row for row in roles['VAL'] if scene(row) in test_scenes]
    roles['VAL'] = [row for row in roles['VAL'] if scene(row) not in test_scenes]
    counts = {role: dict(Counter(row['sensor'] for row in group)) for role, group in roles.items()}
    standard = dict(S2=400, L8=400, L9=400)
    if counts != dict(TRAIN=standard, VAL=dict(S2=100, L8=88, L9=87), TEST=dict(S2=100, L8=100, L9=100)):
        raise ValueError(f'Unexpected PD-HERE member counts: {counts}')
    for field in (scene, 'isolation_group', 'id', 'cache_sha256'):
        index = {}
        for role, group in roles.items():
            for row in group:
                key = field(row) if callable(field) else row[field]
                if not key or index.setdefault(key, role) != role:
                    raise ValueError(f'Cross-role or missing identity: {field} {key}')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate member ID')
    return roles, dict(source_manifest=source['manifest_sha256'], counts=counts,
                       excluded_val=[row['id'] for row in excluded], overlapping_scenes=len({scene(row) for row in excluded}))


@lru_cache(maxsize=8)
def response_table(path):
    return np.genfromtxt(path, delimiter=',', names=True)


class OpticalDataset:
    def __init__(self, root):
        self.root = root
        self.catalog_path = root / 'data/physics/sensor_physics.json'
        self.resolver = MetadataResolver(self.catalog_path)
        self.verified = {}

    @lru_cache(maxsize=128)
    def srf(self, platform, name):
        entry = self.resolver.catalog['sensors'].get(platform, {}).get('srf', {})
        if entry.get('status') not in ('measured', 'derived_gaussian'):
            return Attribute()
        table = response_table(self.catalog_path.parent / entry['csv'])
        if name not in table.dtype.names:
            return Attribute()
        keep = np.isfinite(table[name])
        wavelength, response = table['wavelength_nm'][keep] / 1000., table[name][keep]
        if len(wavelength) < 2 or not (np.diff(wavelength) > 0).all():
            raise ValueError(f'Invalid SRF coordinates: {platform}/{name}')
        gaps = np.diff(wavelength) / 2
        weights = np.r_[gaps, 0.] + np.r_[0., gaps]
        samples = tuple(zip(wavelength.tolist(), [0.] * len(wavelength), response.tolist(), weights.tolist()))
        status = 'observed' if entry['status'] == 'measured' else 'estimated'
        return Attribute(status, 'um', entry['source'], samples=samples)

    def physical(self, row, band, grid):
        catalog_key = {'L8': 'LANDSAT8_OLI', 'L9': 'LANDSAT9_OLI2'}.get(row['sensor'], row['instrument'])
        canonical = re.sub(r'^B0+(\d)', r'B\1', band['band_key'].removeprefix('SR_'))
        rules = self.resolver.catalog['sensors'].get(catalog_key, {})
        record = next((r for r in rules.get('bands', ()) if r['band'] == canonical), {})
        source = f'{self.resolver.catalog_source};{catalog_key}/{canonical}'

        def scalar(field, product_field):
            product = band.get(product_field)
            catalog = record.get(field)
            if product is not None and np.isfinite(product) and product > 0:
                return Attribute('observed', 'um', row['product'], (product / 1000.,))
            if catalog is not None and np.isfinite(catalog) and catalog > 0:
                return Attribute('observed', 'um', source, (catalog / 1000.,))
            return Attribute()

        def spatial(name):
            record = band.get(name) or rules.get(name)
            if not record or record.get('status') not in ('observed', 'measured', 'estimated'):
                return Attribute()
            status = 'estimated' if record['status'] == 'estimated' else 'observed'
            return Attribute(status, record['unit'], record['source'], tuple(record.get('summary', ())),
                             tuple(tuple(p) for p in record.get('samples', ())))

        return Physics(scalar('centre_nm', 'wavelength_nm'), scalar('fwhm_nm', 'fwhm_nm'),
                       self.srf(catalog_key, canonical), Attribute('observed', 'm', 'native affine', grid.spacing),
                       spatial('psf'), spatial('mtf'))

    def load(self, row, device, *, qa_only=False):
        if row['sensor'] not in ('S2', 'L8', 'L9') or row['units'] != 'surface_reflectance':
            raise ValueError(f'Unsupported native reflectance record: {row["id"]}')
        path = self.root / row['cache']
        checked_file(path, row['cache_sha256'], self.verified)
        bands = []
        with np.load(path, allow_pickle=False) as saved:
            for record in row['bands']:
                key = record['band_key']
                valid = saved[key + '__valid']
                values = np.zeros(valid.shape, np.float32) if qa_only else saved[key]
                grid = Grid(tuple(record['shape']), tuple(record['transform']), record['crs'])
                if (values.shape != grid.shape or valid.shape != grid.shape or valid.dtype != np.bool_
                        or values.dtype.kind != 'f' or not np.isfinite(values[valid]).all()):
                    raise ValueError(f'Native cache value, QA or shape mismatch: {row["id"]}/{key}')
                expected = record.get('native_pixel_support_m', grid.spacing)
                if not np.allclose(expected, grid.spacing, rtol=0, atol=1e-6):
                    raise ValueError(f'Native spacing disagrees with affine: {row["id"]}/{key}')
                bands.append(Band(key, torch.as_tensor(values, dtype=torch.float32, device=device),
                                  torch.as_tensor(valid, device=device), grid, self.physical(row, record, grid)))
        if not bands or len({band.key for band in bands}) != len(bands):
            raise ValueError(f'Expected unique bands: {row["id"]}')
        return Observation(row['id'], row['sensor'], tuple(bands))


def task_view(observation, kind, stride, *, seed=17, update=0, panel=None):
    bands = observation.bands
    device = bands[0].values.device
    try:
        work, maps = build_work_grid(tuple(b.grid for b in bands), stride, device)
    except ValueError as error:
        raise ValueError(f'{observation.id}: {error}') from error
    rng = task_rng(seed, update, observation.id, kind)
    if panel is not None:
        fixed = {key: torch.as_tensor(ids, device=device, dtype=torch.long) for key, ids in panel.items()}
        if set(fixed) - {band.key for band in bands}:
            raise ValueError(f'Unknown panel band: {observation.id}/{kind}')
        hidden = []
        for band in bands:
            mask = torch.zeros_like(band.valid)
            if band.key in fixed:
                mask.flatten()[fixed[band.key]] = True
            hidden.append(mask)
    elif kind == 'A':
        h, w = work.shape
        occupied = torch.stack([maps[b.grid].support(b.valid) > 0 for b in bands]).any(0).reshape(h, w)
        padded = torch.nn.functional.pad(occupied, (0, (-w) % 4, 0, (-h) % 4))
        mass = padded.reshape((h + 3) // 4, 4, (w + 3) // 4, 4).sum((1, 3)).flatten().tolist()
        blocks = [i for i, n in enumerate(mass) if n]
        rng.shuffle(blocks)
        mask = torch.zeros_like(work.valid)
        budget, count = sum(mass) * .55, 0
        for block in blocks[:max(1, len(blocks) - 1)]:
            y, x = divmod(block, (w + 3) // 4)
            mask[y * 4:y * 4 + 4, x * 4:x * 4 + 4] = True
            count += mass[block]
            if count >= budget:
                break
        hidden = []
        for band in bands:
            m = maps[band.grid]
            hit = mask.flatten()[m.destination] & (m.area > 0)
            native = torch.zeros(m.pixels, device=device, dtype=torch.int32)
            hidden.append(native.index_add_(0, m.source, hit.int()).reshape(band.grid.shape) > 0)
    elif kind == 'B1':
        eligible = sorted(b.key for b in bands if b.valid.any())
        if len(eligible) < 2:
            return None
        size = min(len(eligible) - 1, max(1, int(len(eligible) * .25 + .5)))
        keys = set(rng.sample(eligible, size))
        hidden = [torch.full_like(b.valid, b.key in keys) for b in bands]
    else:
        raise ValueError(f'Unsupported task: {kind}')

    visible, requests, truth, keys = [], [], [], []
    for band, mask in zip(bands, hidden, strict=True):
        observed = band.valid & ~mask
        mapping = maps[band.grid]
        if observed.any():
            visible.append(VisibleBand(torch.where(observed, band.values, 0.), observed, band.grid, band.physics))
        target = (band.valid & mask).flatten()
        pixels = target.nonzero().flatten()
        if pixels.numel():
            edge = target[mapping.source] & (mapping.area > 0)
            output_index = torch.searchsorted(pixels, mapping.source[edge])
            requests.append(TargetRequest(band.physics, band.grid, pixels, mapping.destination[edge],
                                          output_index, mapping.area[edge] / mapping.pixel_area))
            truth.append(band.values.detach().flatten()[pixels])
            keys.append(band.key)
    if not requests:
        return None
    return TaskView(VisibleObservation(tuple(visible), work), tuple(requests), tuple(truth), tuple(keys))
