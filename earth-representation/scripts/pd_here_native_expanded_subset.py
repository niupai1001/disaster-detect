"""Freeze 2,000 A/B1-capable parents across the core and v6 source distribution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import ssl
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import torch

from encoder_v4.expansion import ExpansionDataset


EXTENSION = Path('outputs/encoder_v4/pd_here_optical_expansion_v6')
EXTENSION_SHA = '66c1fef2132b441b6bc8f88a11d7d45f7cd4324ea0efb6db55d09d0bc91eb28b'
CORE_SHA = '6a7fb668d2a780f49bed27bcde3475c27c193cfc14d3ee7704436736bc7a79cd'
TASKS = {'A', 'B1'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


class ExpandedTrainingData:
    def __init__(self, root):
        folder = root / EXTENSION
        release = read(folder / 'release.json')
        ready = read(folder / 'DATA_READY.json')
        if (digest(folder / 'release.json') != EXTENSION_SHA or ready['release_sha256'] != EXTENSION_SHA
                or ready['status'] != 'DATA_READY' or release['core']['sha256'] != CORE_SHA):
            raise ValueError('Expanded release identity changed')
        self.dataset = ExpansionDataset(folder, root)
        core = self.dataset.core_roles['TRAIN']
        extension = self.dataset.roles['TRAIN']
        self.core = {row['id']: row for row in core}
        self.extension = {row['id']: row for row in extension if set(row['eligible_tasks']) == TASKS}
        if (len(self.core), len(self.extension), len(extension)) != (7908, 5113, 5158):
            raise ValueError('Unexpected release or A/B1 member counts')
        if self.core.keys() & self.extension.keys():
            raise ValueError('Duplicate parent across data releases')

    def members(self):
        core = [dict(id=key, source=f'Core {row["sensor"]}', domain='surface_reflectance', encoding='physical')
                for key, row in self.core.items()]
        extra = [dict(id=key, source=row['dataset'], domain=row['radiometric_domain'],
                      encoding=row.get('value_encoding', 'physical')) for key, row in self.extension.items()]
        return core + extra

    def load(self, row, device):
        key = row['id']
        if key in self.core:
            return self.dataset.core.load(self.core[key], device)
        return self.dataset.load(self.extension[key], device)

    def native_pixels(self, row):
        key = row['id']
        member = self.core[key] if key in self.core else self.extension[key]
        return sum(math.prod(band['shape']) for band in member['bands'])


def allocate(groups, total):
    population = sum(groups.values())
    if not len(groups) <= total <= population or not all(groups.values()):
        raise ValueError('Invalid proportional sample budget')
    quotas = {source: total * count / population for source, count in groups.items()}
    counts = {source: max(1, math.floor(quota)) for source, quota in quotas.items()}
    while sum(counts.values()) < total:
        source = max((s for s in counts if counts[s] < groups[s]), key=lambda s: (quotas[s] - counts[s], s))
        counts[source] += 1
    while sum(counts.values()) > total:
        source = max((s for s in counts if counts[s] > 1), key=lambda s: (counts[s] - quotas[s], s))
        counts[source] -= 1
    return counts


def select_members(members, seed=17):
    groups = defaultdict(list)
    for member in members:
        groups[member['source']].append(member)
    populations = {source: len(group) for source, group in groups.items()}
    counts = allocate(populations, 2000)
    selected = []
    for source, group in sorted(groups.items()):
        domains = defaultdict(list)
        for row in group:
            domains[row['domain']].append(row)
        stratum_counts = allocate({k: len(v) for k, v in domains.items()}, counts[source])
        for domain, stratum in sorted(domains.items()):
            ranked = sorted(stratum, key=lambda row: hashlib.sha256(f'{seed}:{row["id"]}'.encode()).digest())
            selected.extend(ranked[:stratum_counts[domain]])
    return selected


def prepare(root, out):
    data = ExpandedTrainingData(root)
    members = data.members()
    subset = select_members(members)
    scales = {}
    for row in subset:
        if row['encoding'] != 'stored_dn':
            continue
        native = data.load(row, 'cpu')
        maximum = max((band.values[band.valid].abs().amax().item() for band in native.bands if band.valid.any()), default=0.)
        scales[row['domain']] = max(scales.get(row['domain'], 1.), maximum)
    record = dict(schema='pd_here_native_expanded_subset_v1', seed=17, core_release_sha256=CORE_SHA,
                  extension_release_sha256=EXTENSION_SHA, selection='source-proportional 2000; minimum one obs per source/domain; seed17 hash order',
                  population_counts=dict(Counter(row['source'] for row in members)),
                  sample_counts=dict(Counter(row['source'] for row in subset)), excluded_single_task=45,
                  dn_scale_protocol='max absolute QA-valid DN, by product domain, from selected TRAIN parents only',
                  dn_scales=scales, parents=subset, test_locked=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(record, indent=2, allow_nan=False) + '\n'
    if out.exists():
        if out.read_text(encoding='utf-8') != content:
            raise ValueError('Refusing to change frozen expanded TRAIN selection')
    else:
        out.write_text(content, encoding='utf-8')
    print(json.dumps({key: record[key] for key in ('population_counts', 'sample_counts', 'dn_scales')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    prepare(Path.cwd(), args.out)
