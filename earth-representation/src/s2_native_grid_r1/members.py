"""Freeze a random core S2 TRAIN subset against the sealed release."""
import hashlib
import json
import random
from pathlib import Path

RELEASE = Path('outputs/encoder_v4/pd_here_full_optical_v4')
OUTPUT = Path('outputs/s2_native_grid_r1/s2_selection_1000')
RELEASE_SHA = '6a7fb668d2a780f49bed27bcde3475c27c193cfc14d3ee7704436736bc7a79cd'
SEED = 17
COUNT = 1000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(root, output):
    release = root / RELEASE
    raw = (release / 'release.json').read_bytes()
    if sha(raw) != RELEASE_SHA:
        raise ValueError('Unexpected core release identity')
    train = (release / 'train.jsonl').read_bytes()
    if sha(train) != json.loads(raw)['files']['train.jsonl']:
        raise ValueError('TRAIN manifest checksum mismatch')
    s2 = sorted((r for line in train.splitlines() if (r := json.loads(line))['sensor'] == 'S2'),
                key=lambda row: row['id'])
    if len(s2) != 5340 or len({r['id'] for r in s2}) != len(s2):
        raise ValueError('Expected 5340 distinct core S2 parents')
    if any(r['dataset_role'].upper() != 'TRAIN' or r['units'] != 'surface_reflectance' for r in s2):
        raise ValueError('Expected core TRAIN reflectance records')
    rows = random.Random(SEED).sample(s2, COUNT)
    payload = b''.join((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode() for row in rows)
    identity = dict(seed=SEED, sampled=COUNT, eligible=len(s2), sensor='S2', role='TRAIN',
                    method='random.Random(seed).sample on sorted parent IDs, without replacement',
                    release_sha256=RELEASE_SHA, source_train_sha256=sha(train), members_sha256=sha(payload))
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in [('train_1000.jsonl', payload),
                           ('members_identity.json', (json.dumps(identity, indent=2) + '\n').encode())]:
        path = output / name
        if path.exists() and path.read_bytes() != contents:
            raise ValueError(f'Frozen file differs: {path}')
        if not path.exists():
            with path.open('xb') as handle:
                handle.write(contents)
    return rows, identity
