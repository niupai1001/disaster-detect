"""Audit existing optical caches without loading task labels, torch, or download code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / 'outputs/use/external'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def audit_cache(root, row):
    path = (root / row['file']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Cache path escapes source root')
    if sha(path) != row['sha256']:
        raise ValueError('Cache SHA256 changed: ' + row['file'])
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'x', 'pixel_valid', 'metadata', 'known'}:
            raise ValueError('Unexpected cache fields; task labels forbidden')
        x, valid, metadata, known = (arrays[key] for key in ('x', 'pixel_valid', 'metadata', 'known'))
    channels = len(row['band_ids'])
    if x.ndim != 3 or x.shape != valid.shape or x.shape[0] != channels:
        raise ValueError('Image/quality shape mismatch')
    if x.dtype != np.float32 or valid.dtype != np.bool_ or known.dtype != np.bool_:
        raise ValueError('Cache dtype mismatch')
    if metadata.shape != (channels, 3) or known.shape != metadata.shape:
        raise ValueError('Physical descriptor shape mismatch')
    if not np.isfinite(metadata).all() or np.any(metadata[~known] != 0) or np.any(metadata[known] <= 0):
        raise ValueError('Invalid known/unknown physical descriptors')
    if not np.isfinite(x[valid]).all() or not valid.all(axis=0).mean() >= .8:
        raise ValueError('Invalid reflectance or less than 80% common valid pixels')
    if len(set(row['band_ids'])) != channels or not row.get('crs') or len(row['transform']) != 6:
        raise ValueError('Missing band/grid provenance')
    return dict(shape=list(x.shape), common_valid_ratio=float(valid.all(axis=0).mean()),
                reflectance_min=float(x[valid].min()), reflectance_max=float(x[valid].max()))


def members():
    config_path = ROOT / 'outputs/use/pretraining/resolved_config.json'
    config = read(config_path)
    files = [config_path, ROOT/'manifests/v4r1/protocol.json', ROOT/'manifests/v4r1/stats_core25.json']
    pools = {}
    for name, acceptance_name, expected in [
        ('current_manifest.json', 'current_acceptance.json', config['manifest_sha256']),
        ('local_unlabeled_manifest.json', 'local_acceptance.json', config['local_manifest_sha256']),
        ('f_local_manifest.json', 'f_local_acceptance.json', None),
    ]:
        path, acceptance_path = EXTERNAL/name, EXTERNAL/acceptance_name
        acceptance = read(acceptance_path)
        digest = sha(path)
        if acceptance.get('ok') is not True or acceptance['manifest_sha256'] != digest or expected and expected != digest:
            raise ValueError('Source manifest/acceptance mismatch: ' + name)
        if 'protocol_sha256' in acceptance and acceptance['protocol_sha256'] != sha(files[1]):
            raise ValueError('Source event protocol changed')
        for row in read(path):
            if row['file'] in pools:
                raise ValueError('Duplicate source cache identity')
            pools[row['file']] = row
        files.extend([path, acceptance_path])
    paused = EXTERNAL/'acquisition_paused.json'
    if read(paused)['manifest_sha256'] != config['manifest_sha256']:
        raise ValueError('Paused acquisition snapshot differs')
    files.append(paused)
    result = []
    seen = set()
    def append(ids, role):
        for name in ids:
            if name in seen or name not in pools:
                raise ValueError('Missing or overlapping role member: ' + name)
            seen.add(name)
            result.append(dict(pools[name], role=role))
    for key, role in [('train_ids', 'TRAIN'), ('validation_ids', 'PRETRAIN_VAL')]:
        for names in config[key].values():
            append(names, role)
    f_acceptance = read(EXTERNAL/'f_local_acceptance.json')
    for key, role in [('train_ids', 'TRAIN'), ('validation_ids', 'PRETRAIN_VAL'), ('guard_ids', 'GUARD')]:
        append(f_acceptance[key], role)
    if seen != set(pools):
        raise ValueError('Unassigned source members')
    return result, {str(path.relative_to(ROOT)): sha(path) for path in files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/use_independent_frozen/cache_audit_v1')
    args = parser.parse_args()
    rows, source_hashes = members()
    args.output.mkdir(parents=True, exist_ok=False)
    failures = []
    with (args.output/'members.jsonl').open('x', encoding='utf8') as handle:
        for index, row in enumerate(rows):
            record = dict(row)
            try:
                record['cache_audit'] = audit_cache(EXTERNAL, row)
            except (ValueError, OSError, KeyError) as error:
                record['error'] = str(error)
                failures.append(dict(file=row['file'], error=str(error)))
            handle.write(json.dumps(record, ensure_ascii=False)+'\n')
            if (index+1) % 500 == 0:
                handle.flush()
                print(f'{index+1}/{len(rows)}; failures={len(failures)}', flush=True)
    report = dict(ok=not failures, samples=len(rows), roles=dict(Counter(row['role'] for row in rows)),
                  source_sha256=source_hashes, members_sha256=sha(args.output/'members.jsonl'),
                  failures=failures, limitation='Reuses accepted geographic/radiometric source audits; does not re-read source products or establish a new geography test set')
    (args.output/'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({key: report[key] for key in ('ok', 'samples', 'roles', 'failures')}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
