import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import yaml
from check_research_protocol import audit_config


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit(config):
    cfg=yaml.safe_load(Path(config).read_text());v=cfg['use_next_p2']
    expected=6000 if v['task']=='pretrain' else 3000
    assert v['stage']=='P2' and v['mode'] in ('pixel','physical') and v['task'] in ('pretrain','head')
    assert v['steps']==cfg['segformer']['train']['max_steps']==v['warmup_updates']*20==expected
    assert (v['width'],v['layers'],v['heads'],v['physical_patch_m'],v['subcell_m'],v['pixel_block'])==(192,2,6,120,30,4)
    assert v['microbatch']==2 and v['effective_batch']==8 and v['seed']==1234 and v['head_seed']==1235
    assert v['tf32'] is False and v['precision']=='FP32'
    assert v['selection']==('fixed_u6000' if v['task']=='pretrain' else 'dev_event_mean_iou')
    assert cfg['segformer']['optim']['encoder_lr']==(.0003 if v['task']=='pretrain' else 0.)
    result=audit_config(config)
    release=ROOT/v['data_release'];contract=json.loads((release/'release.json').read_text())
    assert contract['ok'] and sha(release/'manifest.jsonl')==v['data_manifest_sha256']==contract['manifest_sha256']
    assert contract['legacy_validation']==599
    calibration=json.loads((ROOT/v['calibration']).read_text())
    rows={r['id']:r for r in (json.loads(l) for l in (release/'manifest.jsonl').read_text().splitlines())}
    for stats in calibration.values():
        assert stats['fit_role']=='TRAIN' and all(rows[i]['role']=='TRAIN' for i in stats['fit_ids'])
        assert all(s>0 for s in stats['std'])
    files={name:sha(ROOT/v[name]) for name in ('calibration','schedule','u0')}
    result['use_next_p2']=dict(ok=True,mode=v['mode'],task=v['task'],steps=expected,bindings=files,
                             data_manifest_sha256=contract['manifest_sha256'],ssl_counts=contract['counts'],formal_updates_started=0)
    result['stage_contract']=result['use_next_p2']
    result['reference_supervised_selection']=result['selection']
    result['selection']=v['selection']
    result['actual_training_role']='TRAIN_unlabeled' if v['task']=='pretrain' else 'TRAIN_supervised'
    return result


if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2))
