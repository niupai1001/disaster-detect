"""Frozen optical members, physical metadata, and value-isolated task views."""
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
import json
import math
from pathlib import Path, PureWindowsPath
import re

import numpy as np
import torch

from encoder_v3.data import checked_file, sha256
from encoder_v3.shared_mae.metadata import MetadataResolver
from encoder_v3.tasks import task_rng

from .geometry import Grid, WorkGrid, build_work_grid
from .release import SCHEMA as RELEASE_SCHEMA, acquisition, load_release, read_json


@dataclass(frozen=True)
class Attribute:
    status: str = 'missing'
    unit: str = ''
    source: str = ''
    summary: tuple[float, ...] = ()
    samples: tuple[tuple[float, float, float, float], ...] = ()  # x, y/direction, response, weight
    summary_fields: tuple[str, ...] = ()
    _hash: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.status not in ('missing', 'observed', 'estimated', 'not_applicable', 'withheld'):
            raise ValueError(f'Unexpected attribute state: {self.status}')
        if self.status in ('observed', 'estimated'):
            if not self.source or not self.unit or not (self.summary or self.samples):
                raise ValueError('Available physical attributes require values, units and provenance')
        elif self.summary or self.samples or self.summary_fields:
            raise ValueError('Unavailable physical attributes cannot carry observations')
        if (not np.isfinite(self.summary).all() or any(len(p) != 4 for p in self.samples)
                or not np.isfinite(self.samples).all() or any(p[3] <= 0 for p in self.samples)):
            raise ValueError('Nonfinite or malformed physical samples')
        object.__setattr__(self, '_hash', hash((self.status, self.unit, self.source, self.summary,
                                                self.samples, self.summary_fields)))

    def __hash__(self):
        return self._hash


UNITS = {
    'center': ('um', {'um': 1., 'µm': 1., 'nm': .001}),
    'bandwidth': ('um', {'um': 1., 'µm': 1., 'nm': .001}),
    'srf': ('um', {'um': 1., 'µm': 1., 'nm': .001}),
    'gsd': ('m', {'m': 1., 'cm': .01, 'km': 1000.}),
    'psf': ('m', {'m': 1., 'cm': .01, 'km': 1000.}),
    'mtf': ('cycles/m', {'cycles/m': 1., 'cycles/km': .001}),
}


@lru_cache(maxsize=8192)
def canonical_attribute(name, attr):
    if attr.status not in ('observed', 'estimated'):
        return attr
    unit, choices = UNITS[name]
    if attr.unit not in choices:
        raise ValueError(f'{name}: unsupported unit {attr.unit!r}')
    factor = choices[attr.unit]
    schema = {'center': ('wavelength',), 'bandwidth': ('fwhm',), 'gsd': ('dx', 'dy'),
              'psf': ('sigma_x', 'sigma_y', 'direction_rad'),
              'mtf': ('frequency', 'response', 'direction_rad')}.get(name, ())
    if attr.summary:
        count = len(attr.summary)
        if name in ('psf', 'mtf'):
            if count not in (2, 3) or attr.summary_fields != schema[:count]:
                raise ValueError(f'{name}: summary fields must be {schema[:2]} or {schema}')
        elif count != len(schema) or not schema or attr.summary_fields not in ((), schema):
            raise ValueError(f'{name}: summary requires fields {schema}')
    elif attr.summary_fields or name in ('center', 'bandwidth', 'gsd'):
        raise ValueError(f'{name}: scalar summary required')
    summary = list(attr.summary)
    if name in ('center', 'bandwidth', 'gsd', 'psf'):
        size = min(len(summary), 2)
        if any(v <= 0 for v in summary[:size]):
            raise ValueError(f'{name}: distances and widths must be positive')
        summary[:size] = [v * factor for v in summary[:size]]
    if name == 'mtf' and summary:
        if summary[0] < 0 or not 0 <= summary[1] <= 1:
            raise ValueError('mtf: invalid frequency or response')
        summary[0] *= factor
    if len(summary) == 3 and not -math.pi <= summary[2] <= math.pi:
        raise ValueError(f'{name}: direction must be in radians in [-pi, pi]')
    if attr.samples and name not in ('srf', 'psf', 'mtf'):
        raise ValueError(f'{name}: use a scalar summary')
    samples = []
    for x, y, response, weight in attr.samples:
        if response < 0 or (name in ('srf', 'mtf') and response > 1.000001):
            raise ValueError(f'{name}: invalid response')
        if (name == 'srf' and (x <= 0 or y != 0)
                or name == 'mtf' and (x < 0 or not -math.pi <= y <= math.pi)):
            raise ValueError(f'{name}: invalid sampling coordinates')
        samples.append((round(x * factor, 15), y * factor if name == 'psf' else y,
                        response, weight * factor ** (2 if name == 'psf' else 1)))
    if name == 'srf' and any(a[0] >= b[0] for a,b in zip(samples,samples[1:])):
        raise ValueError('srf: wavelengths must increase')
    return Attribute(attr.status, unit, attr.source, tuple(summary), tuple(samples),
                     schema[:len(summary)])


@dataclass(frozen=True)
class Physics:
    center: Attribute = Attribute()
    bandwidth: Attribute = Attribute()
    srf: Attribute = Attribute()
    gsd: Attribute = Attribute()
    psf: Attribute = Attribute()
    mtf: Attribute = Attribute()

    def __post_init__(self):
        for name in UNITS:
            object.__setattr__(self, name, canonical_attribute(name, getattr(self, name)))


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
    wrong_physics: Physics | None = None


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
    wrong_physics: Physics | None = None


@dataclass(frozen=True)
class TaskView:
    visible: VisibleObservation
    requests: tuple[TargetRequest, ...]
    truth: tuple[torch.Tensor, ...]
    keys: tuple[str, ...]
    visible_keys: tuple[str, ...] = ()
    curve_targets: tuple = ()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def asset_path(root, name):
    native = Path(name)
    if native.is_absolute():
        return native
    windows = PureWindowsPath(name)
    if windows.drive:
        if len(windows.drive) == 2 and windows.drive.endswith(':'):
            return Path('/mnt') / windows.drive[0].lower() / Path(*windows.parts[1:])
        raise ValueError(f'Use an accessible mounted path for {name}')
    return root.joinpath(*windows.parts)


def scene(row):
    return tuple(acquisition(row['acquisition_key']).split(':'))


def derive_members(release):
    source = read_json(release / 'release.json')
    if source.get('schema') == RELEASE_SCHEMA:
        return load_release(release)
    # Frozen comparison publications retain their original derived panels.
    if sha256(release / 'members.jsonl') != source['manifest_sha256']:
        raise ValueError('Research release manifest hash mismatch')
    rows = read_rows(release / 'members.jsonl')
    roles = {role: [row for row in rows if row['dataset_role'] == role] for role in ('TRAIN', 'VAL', 'TEST')}
    if sum(map(len, roles.values())) != len(rows):
        raise ValueError('Unexpected research release member count or role')
    test_scenes = {scene(row) for row in roles['TEST']}
    excluded = [row for row in roles['VAL'] if scene(row) in test_scenes]
    roles['VAL'] = [row for row in roles['VAL'] if scene(row) not in test_scenes]
    counts = {role: dict(Counter(row['sensor'] for row in group)) for role, group in roles.items()}
    if not all(roles.values()):
        raise ValueError('Each frozen comparison role must be nonempty')
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
            summary = record.get('summary', {})
            if not isinstance(summary, dict):
                raise ValueError(f'{name}: use named summary fields')
            fields = (('sigma_x', 'sigma_y', 'direction_rad') if name == 'psf'
                      else ('frequency', 'response', 'direction_rad'))
            if set(summary) - set(fields):
                raise ValueError(f'{name}: unknown summary fields')
            names = tuple(k for k in fields if k in summary)
            return Attribute(status, record['unit'], record['source'], tuple(summary[k] for k in names),
                             tuple(tuple(p) for p in record.get('samples', ())), names)

        return Physics(scalar('centre_nm', 'wavelength_nm'), scalar('fwhm_nm', 'fwhm_nm'),
                       self.srf(catalog_key, canonical), Attribute('observed', 'm', 'native affine', grid.spacing),
                       spatial('psf'), spatial('mtf'))

    def load(self, row, device, *, qa_only=False):
        if row['sensor'] not in ('S2', 'L8', 'L9') or row['units'] != 'surface_reflectance':
            raise ValueError(f'Unsupported native reflectance record: {row["id"]}')
        path = asset_path(self.root, row['cache'])
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
                bands.append(Band(key, torch.as_tensor(values, dtype=torch.float32),
                                  torch.as_tensor(valid), grid, self.physical(row, record, grid)))
        if not bands or len({band.key for band in bands}) != len(bands):
            raise ValueError(f'Expected unique bands: {row["id"]}')
        return observation_to(Observation(row['id'], row['sensor'], tuple(bands)), device)


def observation_to(observation, device):
    if observation.bands[0].values.device == torch.device(device):
        return observation
    groups = defaultdict(list)
    bands = list(observation.bands)
    for index, band in enumerate(bands):
        groups[band.grid.shape].append(index)
    for indices in groups.values():
        values = torch.stack([bands[i].values for i in indices]).to(device, non_blocking=True)
        valid = torch.stack([bands[i].valid for i in indices]).to(device, non_blocking=True)
        for i, x, m in zip(indices, values, valid, strict=True):
            bands[i] = replace(bands[i], values=x, valid=m)
    return replace(observation, bands=tuple(bands))


def task_view(observation, kind, stride, *, seed=17, update=0, panel=None, skip=None,
              target_support=None, allow_empty_visible=False):
    def unavailable(reason):
        if skip is not None:
            skip.append(dict(id=observation.id, task=kind, reason=reason))
        return None

    if kind not in ('A', 'B1'):
        raise ValueError(f'Unsupported task: {kind}')
    bands = observation.bands
    device = bands[0].values.device
    try:
        work, maps = build_work_grid(tuple(b.grid for b in bands), stride, device)
    except ValueError as error:
        raise ValueError(f'{observation.id}: {error}') from error
    rng = task_rng(seed, update, observation.id, kind)
    if panel is not None:
        fixed = {key: np.asarray(ids, dtype=np.int64) for key, ids in panel.items()}
        if set(fixed) - {band.key for band in bands}:
            raise ValueError(f'Unknown panel band: {observation.id}/{kind}')
        hidden = []
        for band in bands:
            mask = torch.zeros_like(band.valid)
            if band.key in fixed:
                ids = fixed[band.key]
                if ids.ndim != 1 or np.any((ids < 0) | (ids >= mask.numel())):
                    raise ValueError(f'Out-of-range native target: {observation.id}/{kind}/{band.key}')
                mask.flatten()[torch.as_tensor(ids, device=device)] = True
                if kind == 'B1' and not torch.equal(mask & band.valid, band.valid):
                    raise ValueError(f'B1 panel must remove the whole valid band: {observation.id}/{band.key}')
            hidden.append(mask)
    elif kind == 'A':
        h, w = work.shape
        groups = defaultdict(list)
        for b in bands:
            groups[b.grid].append(b.valid)
        occupied = torch.stack([maps[g].support(torch.stack(valid).any(0)) > 0
                                for g, valid in groups.items()]).any(0).reshape(h, w)
        padded = torch.nn.functional.pad(occupied, (0, (-w) % 4, 0, (-h) % 4))
        mass = padded.reshape((h + 3) // 4, 4, (w + 3) // 4, 4).sum((1, 3)).flatten().tolist()
        blocks = [i for i, n in enumerate(mass) if n]
        rng.shuffle(blocks)
        budget, count = sum(mass) * .55, 0
        selected = [False] * len(mass)
        for block in blocks[:max(1, len(blocks) - 1)]:
            selected[block] = True
            count += mass[block]
            if count >= budget:
                break
        mask = torch.tensor(selected, device=device, dtype=torch.bool).reshape((h + 3) // 4, (w + 3) // 4)
        mask = mask.repeat_interleave(4, 0).repeat_interleave(4, 1)[:h, :w]
        by_grid = {}
        for grid, m in maps.items():
            hit = mask.flatten()[m.destination] & (m.area > 0)
            native = torch.zeros(m.pixels, device=device, dtype=torch.int32)
            by_grid[grid] = native.index_add_(0, m.source, hit.int()).reshape(grid.shape) > 0
        hidden = [by_grid[b.grid] for b in bands]
    elif kind == 'B1':
        eligible = sorted(b.key for b, ok in zip(bands, torch.stack([b.valid.any() for b in bands]).tolist())
                          if ok)
        if len(eligible) < 2:
            return unavailable('fewer than two QA-valid bands')
        size = min(len(eligible) - 1, max(1, int(len(eligible) * .25 + .5)))
        keys = set(rng.sample(eligible, size))
        hidden = [torch.full_like(b.valid, b.key in keys) for b in bands]

    visible, requests, truth, keys, visible_keys = [], [], [], [], []
    observed_masks = [b.valid & ~mask for b, mask in zip(bands, hidden, strict=True)]
    visible_flags = torch.stack([mask.any() for mask in observed_masks]).tolist()
    if not any(visible_flags) and not allow_empty_visible:
        return unavailable('no visible QA support after native footprint masking')
    targets = [b.valid & mask for b, mask in zip(bands, hidden, strict=True)]
    if target_support is not None:
        targets = [target & target_support[b.key] if b.key in target_support else torch.zeros_like(target)
                   for b, target in zip(bands, targets, strict=True)]
    target_flags = torch.stack([mask.any() for mask in targets]).tolist()
    physical = sorted(bands, key=lambda b: b.key)
    shift = 1 + task_rng(seed, 0, observation.id, 'wrong-physics').randrange(len(bands) - 1) if len(bands) > 1 else 0
    alternate = {b.key: physical[(i + shift) % len(bands)].physics for i, b in enumerate(physical)} if shift else {}
    for band, observed, active, target, wanted in zip(bands, observed_masks, visible_flags, targets, target_flags):
        mapping = maps[band.grid]
        if active:
            visible.append(VisibleBand(torch.where(observed, band.values, 0.), observed, band.grid,
                                       band.physics, alternate.get(band.key)))
            visible_keys.append(band.key)
        if wanted:
            target = target.flatten()
            pixels = target.nonzero().flatten()
            edge = target[mapping.source] & (mapping.area > 0)
            output_index = torch.searchsorted(pixels, mapping.source[edge])
            requests.append(TargetRequest(band.physics, band.grid, pixels, mapping.destination[edge],
                                          output_index, mapping.area[edge] / mapping.pixel_area, alternate.get(band.key)))
            truth.append(band.values.detach().flatten()[pixels])
            keys.append(band.key)
    if not requests:
        return unavailable('no QA-valid native target after footprint masking')
    return TaskView(VisibleObservation(tuple(visible), work), tuple(requests), tuple(truth), tuple(keys),
                    tuple(visible_keys))
