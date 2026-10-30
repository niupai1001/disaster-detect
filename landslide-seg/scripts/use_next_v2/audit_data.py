"""Read-only cache audit of the fixed v2 membership; never follows active_release."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'outputs/use_next_v2/planning/data_snapshot'
OUT = ROOT / 'outputs/use_next_v2/data'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(row):
    result = {'id': row['id'], 'role': row['role'], 'sensor': row['sensor'], 'reasons': []}
    try:
        if sha(row['cache_uri']) != row['sha256']:
            raise ValueError('cache_sha256_mismatch')
        with np.load(row['cache_uri'], allow_pickle=False) as a:
            x, valid = a['x'], a['pixel_valid']
            if x.ndim != 3 or x.shape[0] != len(row['band_ids']) or valid.shape != x.shape:
                raise ValueError('array_channel_or_mask_shape')
            if not np.isfinite(x[valid]).all() or not valid.any():
                raise ValueError('nonfinite_or_empty_valid')
            result.update(shape=list(x.shape), valid_ratio=float(valid.mean()),
                          minimum=float(x[valid].min()), maximum=float(x[valid].max()))
        t = row.get('transform')
        if not row.get('crs') or not t or len(t) != 6 or abs(t[0]*t[4]-t[1]*t[3]) < 1e-10:
            result['reasons'].append('unknown_or_singular_geometry')
        if not row.get('date') or not row.get('units'):
            result['reasons'].append('missing_date_or_units')
        result['native_resolution_known'] = bool(row.get('native_resolution'))
        result['effective_resolution_known'] = bool(row.get('effective_resolution'))
    except Exception as e:
        result['reasons'].append(f'{type(e).__name__}: {e}')
    return result


def pair_audit(p, rows):
    reasons = []
    views = p.get('views', {'s2': p.get('s2_id'), 'landsat': p.get('landsat_id')})
    members = [rows[v] for v in views.values() if v in rows]
    if len(members) != len(views) or len(members) < 2:
        reasons.append('missing_members')
    if any(r['role'] != p['role'] for r in members):
        reasons.append('role_mismatch')
    try:
        dates = [datetime.fromisoformat(r['date'].replace('Z', '+00:00')).date() for r in members]
        if (max(dates)-min(dates)).days > 30:
            reasons.append('date_difference_over_30_days')
    except (KeyError, ValueError):
        reasons.append('date_unknown')
    if p.get('joint_valid_ratio', 0) < .8:
        reasons.append('joint_support_below_80_percent')
    # Published spatial qualification is not evidence of unchanged scene/registration.
    if not p.get('change_screen_passed'):
        reasons.append('change_screen_unverified')
    if not p.get('geometry_screen_passed'):
        reasons.append('geometry_screen_unverified')
    return {'id': p['id'], 'role': p['role'], 'eligible': not reasons, 'reasons': reasons,
            'dense_sar_target_eligible': False}


def main():
    identity = json.loads((SNAPSHOT/'snapshot_identity.json').read_text())
    for name in ('manifest', 'pairs'):
        assert sha(SNAPSHOT/f'{name}.jsonl') == identity[f'{name}_sha256'], name
    rows = [json.loads(l) for l in (SNAPSHOT/'manifest.jsonl').read_text().splitlines()]
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        audited = list(pool.map(audit, rows))
    (OUT/'asset_audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audited))
    pairs = [pair_audit(json.loads(l), {r['id']: r for r in rows})
             for l in (SNAPSHOT/'pairs.jsonl').read_text().splitlines()]
    (OUT/'pair_eligibility.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    report = dict(source_identity=identity, audited=len(rows),
                  counts=dict(Counter(r['role']+'/'+r['sensor'] for r in rows)),
                  legacy_validation=sum(r['role']=='PRETRAIN_VAL' and r['id'].startswith('old_') for r in rows),
                  failed=sum(bool(r['reasons']) for r in audited),
                  failures=[r for r in audited if r['reasons']],
                  pair_eligible=sum(p['eligible'] for p in pairs),
                  native_resolution_unknown=sum(not r.get('native_resolution_known', False) for r in audited),
                  status='AUDITED_NOT_V2_ADMITTED', formal_updates=0)
    (OUT/'audit_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('failures','source_identity')}))


if __name__ == '__main__':
    main()
