"""P1-specific declarations checked in addition to the unchanged event protocol."""
import hashlib
import json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(cfg,root):
    s=cfg['segformer'];v=cfg['use_next_v2']
    if v['stage']!='P1' or v['anchor'] not in ('dofa','anysat'):raise ValueError('Unsupported v2 stage/anchor')
    if s['train']['max_steps']!=3000 or v['schedule_updates']!=3000 or v['warmup_updates']!=150:
        raise ValueError('P1 requires fixed 3000-update cosine with 150 warmup')
    if v['effective_batch']!=8 or v['microbatch'] not in (1,2):raise ValueError('P1 common capacity contract mismatch')
    if v['precision']!='FP32' or v['tf32'] is not False or v['head_seed']!=1235:raise ValueError('P1 initialization/precision mismatch')
    release=root/v['data_release'];data=json.loads((release/'release.json').read_text())
    if not data['ok'] or sha(release/'manifest.jsonl')!=data['manifest_sha256']:raise ValueError('Changed v2 data release')
    if v['data_manifest_sha256']!=data['manifest_sha256']:raise ValueError('P1 points to different v2 cohort')
    if data['legacy_validation']!=599:raise ValueError('Original validation599 changed')
    for path,expected in v['weights'].items():
        if sha(root/path)!=expected:raise ValueError('Anchor checkpoint identity mismatch')
    return dict(ok=True,stage='P1',anchor=v['anchor'],data_manifest_sha256=data['manifest_sha256'],
                formal_start_requires='matching independent READY_P0 and P1 authorization',formal_started=False)
