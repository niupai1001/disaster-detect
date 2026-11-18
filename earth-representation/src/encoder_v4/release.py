"""Sealed optical data release, native profile checks and cross-role isolation."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import math
import re

from rasterio.crs import CRS
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box
from shapely.strtree import STRtree

SCHEMA = 'pd_here_optical_release_v1'
ROLES = ('TRAIN', 'VAL', 'TEST')
S2 = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')
SR7 = tuple(f'SR_B{i}' for i in range(1, 8))
PROFILES = {
    'S2_L2A_CORE10': S2,
    'S2_L2A_4BAND': ('B02', 'B03', 'B04', 'B08'),
    'S2_L2A_12BAND': ('B01', *S2[:-2], 'B09', *S2[-2:]),
    **{f'{sensor}_C2L2_SR{len(bands)}': bands
       for sensor in ('L8', 'L9') for bands in (SR7, SR7[1:])},
}
POLICY = dict(min_band_valid=0.5, acquired_holdout_min_valid=0.95,
              heldout_buffer_degrees=0.01, s2_bad_classes=[0, 1, 3, 8, 9, 10, 11, 255],
              landsat_bad_bits=63, landsat_radsat=0, value_space='surface_reflectance')


def file_sha(path):
    with Path(path).open('rb') as handle:
        return sha256_handle(handle)


def sha256_handle(handle):
    digest = sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b''):
        digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def acquisition(product, date=None):
    """Strip provider aliases and processing dates from acquisition identity."""
    if not isinstance(product, str):
        raise ValueError('No product or acquisition identifier')
    if match := re.fullmatch(r'(S2[A-Z]|LC0[89]):(\d{8}):([0-9A-Z]+)', product):
        return ':'.join((match[1], match[2], match[3].zfill(5)))
    if match := re.match(r'(S2[A-Z])_MSIL2A_(\d{8})T\d{6}(?:_N\d+)?_R\d+_T(\d{2}[A-Z]{3})(?:_|$)', product):
        return ':'.join(match.groups())
    if match := re.match(r'(S2[A-Z])_(\d{1,2}[A-Z]{3})_(\d{8})_\d+_L2A', product):
        return ':'.join((match[1], match[3], match[2].zfill(5)))
    if match := re.match(r'(LC0[89])_L[12]\w\w_(\d{6})_(\d{8})(?:_|$)', product):
        return ':'.join((match[1], match[3], match[2]))
    raise ValueError(f'Unsupported acquisition identity {product}')


def optical_identity(row):
    from_product = acquisition(row['product'])
    from_key = acquisition(row.get('acquisition_key') or row['product'])
    if from_product != from_key:
        raise ValueError('Product and acquisition identifiers disagree')
    date = str(row.get('date', ''))[:10].replace('-', '')
    if date != from_product.split(':')[1]:
        raise ValueError('Product and observation dates disagree')
    sensor = {'LC08': 'L8', 'LC09': 'L9'}.get(from_product[:4], 'S2')
    if sensor != row['sensor']:
        raise ValueError('Acquisition sensor disagrees with profile')
    return from_product


def profile(row):
    keys = [band['band_key'] for band in row['bands']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate band')
    for name, expected in PROFILES.items():
        if name.startswith(row['sensor'] + '_') and set(keys) == set(expected):
            return name
    raise ValueError('Incomplete or unsupported native band set')


def validate_native(row, ordered=True):
    name = profile(row)
    if row.get('dataset_profile', name) != name or row['units'] != 'surface_reflectance':
        raise ValueError('Profile or physical quantity mismatch')
    if ordered and tuple(band['band_key'] for band in row['bands']) != PROFILES[name]:
        raise ValueError('Noncanonical band order')
    footprint, crs = [], None
    for band in row['bands']:
        key = band['band_key']
        gsd = (60 if key in ('B01', 'B09') else
               10 if key in PROFILES['S2_L2A_4BAND'] else 20) if row['sensor'] == 'S2' else 30
        offset = float(band['offset'])
        scale = 0.0001 if row['sensor'] == 'S2' else 0.0000275
        if not math.isclose(float(band['scale']), scale, rel_tol=1e-6) or not any(
            math.isclose(offset, option, abs_tol=1e-7)
            for option in ((0., -0.1) if row['sensor'] == 'S2' else (-0.2,))
        ):
            raise ValueError(f'Unsupported calibration {key}')
        transform = tuple(band['transform'])
        shape = band['shape']
        projection = CRS.from_user_input(band['crs'])
        if crs is None:
            crs = projection
        if (crs != projection or not crs.is_projected or crs.linear_units_factor[1] != 1
                or len(transform) != 6 or not all(map(math.isfinite, transform))
                or not math.isclose(transform[0], gsd) or not math.isclose(transform[4], -gsd)
                or transform[1] or transform[3] or len(shape) != 2
                or any(not isinstance(x, int) or x < 1 for x in shape)
                or tuple(band.get('native_pixel_support_m', (gsd, gsd))) != (gsd, gsd)):
            raise ValueError(f'Unsupported native geometry {key}')
        footprint.append(array_bounds(*shape, Affine(*transform)))
    footprint = tuple(f(box_[i] for box_ in footprint) for i, f in enumerate((min, min, max, max)))
    return tuple(transform_bounds(crs, 'EPSG:4326', *footprint))


def window_key(scene, bounds):
    return f'{scene}:' + ','.join(f'{x:.5f}' for x in bounds)


def identity_tokens(row):
    result = {('id', row['id']), ('window', row['observation_key']),
              ('cache', row['cache_sha256']), ('scene', row['canonical_acquisition'])}
    result.update(('scene', alias) for alias in row.get('acquisition_aliases', ()))
    for field in ('isolation_group', 'source_cache_sha256', 'completed_from_sha256', 'anchor_id'):
        if row.get(field):
            result.add((field, row[field]))
    return result


class Isolation:
    def __init__(self, members, pad):
        self.keys = {}
        self.members = list(members)
        self.tree = STRtree([box(*row['native_bbox']) for row in self.members])
        self.pad = pad
        for row in self.members:
            for key in identity_tokens(row):
                self.keys.setdefault(key, row['id'])

    def conflict(self, row):
        for key in sorted(identity_tokens(row)):
            if key in self.keys:
                return key[0], self.keys[key]
        left, bottom, right, top = row['native_bbox']
        extended = box(left - self.pad, bottom - self.pad, right + self.pad, top + self.pad)
        hits = self.tree.query(extended, predicate='intersects')
        if len(hits):
            return 'footprint_buffer', self.members[min(hits)]['id']
        return None


def check_splits(roles, pad):
    for key in ('id', 'observation_key', 'cache_sha256'):
        values = [row[key] for role in ROLES for row in roles[role]]
        if len(values) != len(set(values)):
            raise ValueError(f'Duplicate release member {key}')
    for index, role in enumerate(ROLES[:-1]):
        other = Isolation((row for name in ROLES[index+1:] for row in roles[name]), pad)
        for row in roles[role]:
            if conflict := other.conflict(row):
                raise ValueError(f'Cross-role {conflict[0]} conflict: {row["id"]}/{conflict[1]}')


def load_release(folder):
    folder = Path(folder)
    release = read_json(folder / 'release.json')
    if release['schema'] != SCHEMA or release['policy'] != POLICY or release['profiles'] != {
        key: list(bands) for key, bands in PROFILES.items()
    }:
        raise ValueError('Unsupported optical release schema or policy')
    required = {'assets.jsonl', 'observations.jsonl', 'members.jsonl', 'quality.jsonl',
                'extension.json', 'inputs.json', *(f'{role.lower()}.jsonl' for role in ROLES)}
    if not required <= release['files'].keys():
        raise ValueError('Incomplete optical release')
    for name, digest in release['files'].items():
        if Path(name).name != name or file_sha(folder / name) != digest:
            raise ValueError(f'Release file hash mismatch: {name}')
    quality_rows = rows(folder / 'quality.jsonl')
    quality = {q['id']: q for q in quality_rows}
    roles = {role: rows(folder / f'{role.lower()}.jsonl') for role in ROLES}
    if any(not values for values in roles.values()):
        raise ValueError('Release requires nonempty TRAIN, VAL and TEST')
    combined = sum(roles.values(), [])
    if (combined != rows(folder / 'members.jsonl') or len(quality) != len(quality_rows)
            or set(quality) != {row['id'] for row in combined}):
        raise ValueError('Quality and member manifests disagree')
    for role, values in roles.items():
        for row in values:
            bbox = validate_native(row)
            q = quality[row['id']]
            threshold = POLICY['acquired_holdout_min_valid'] if row.get('high_qa') else POLICY['min_band_valid']
            if (row['dataset_role'] != role or row['canonical_acquisition'] != optical_identity(row)
                    or row['observation_key'] != window_key(row['canonical_acquisition'], bbox)
                    or list(bbox) != row['native_bbox'] or row['cache_sha256'] != q['cache_sha256']
                    or q['method'] not in ('raw_pixel_QA', 'bound_source_pixel_mask')
                    or set(q['valid_fraction']) != set(PROFILES[row['dataset_profile']])
                    or any(not threshold <= value <= 1 for value in q['valid_fraction'].values())
                    or not q['evidence']):
                raise ValueError(f'Invalid sealed member or quality: {row["id"]}')
    check_splits(roles, POLICY['heldout_buffer_degrees'])
    if release['counts'] != {role: dict(Counter(row['sensor'] for row in values)) for role, values in roles.items()}:
        raise ValueError('Release counts disagree')
    return roles, dict(schema=SCHEMA, source_manifest=release['files']['members.jsonl'],
                       release_sha256=file_sha(folder / 'release.json'), counts=release['counts'],
                       value_space=POLICY['value_space'])

