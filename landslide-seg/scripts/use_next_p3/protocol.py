import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import yaml
from check_research_protocol import audit_config
from use_next_p2.protocol import sha


def audit(config):
    cfg=yaml.safe_load(Path(config).read_text());v=cfg['use_next_p3'];result=audit_config(config)
    assert v['objective'] in ('scale','latent','joint') and v['mode']=='physical'
    assert v['task'] in ('pretrain','head') and v['stage']==('P3' if v['task']=='pretrain' else 'P5')
    assert v['steps']==cfg['segformer']['train']['max_steps']==(10000 if v['task']=='pretrain' else 3000)
    assert v['warmup_updates']==v['steps']//20 and v['effective_batch']==8 and v['microbatch']==2
    assert (v['width'],v['layers'],v['heads'],v['physical_patch_m'],v['subcell_m'])==(192,2,6,120,30)
    assert v['selection']==('fixed_u10000' if v['task']=='pretrain' else 'dev_event_mean_iou')
    assert v['head_seed']==1235 and v['seed']==1234 and v['precision']=='FP32' and v['tf32'] is False
    assert cfg['segformer']['optim']['encoder_lr']==(.0003 if v['task']=='pretrain' else 0.)
    assert v['source_objective']==v['objective']
    assert v['band_drop']==dict(probability=.5,minimum_fraction=.2,maximum_fraction=.4,integer_policy='clamp_round_to_ceil_floor')
    assert v['ema']==dict(start=.996,end=1.,schedule='cosine',boundary='new local encoder only; anchors excluded')
    release=ROOT/v['data_release'];data=json.loads((release/'release.json').read_text())
    assert data['ok'] and sha(release/'manifest.jsonl')==data['manifest_sha256']==v['data_manifest_sha256']
    rows={r['id']:r for r in (json.loads(l) for l in (release/'manifest.jsonl').read_text().splitlines())}
    for stats in json.loads((ROOT/v['calibration']).read_text()).values():
        assert stats['fit_role']=='TRAIN' and all(rows[i]['role']=='TRAIN' for i in stats['fit_ids'])
    result['stage_contract']=dict(stage=v['stage'],objective=v['objective'],task=v['task'],
                                 bindings={k:sha(ROOT/v[k]) for k in ('u0','schedule','calibration')})
    result['reference_supervised_selection']=result['selection']
    result['selection']=v['selection']
    result['actual_training_role']='TRAIN_unlabeled' if v['task']=='pretrain' else 'TRAIN_supervised'
    return result


if __name__=='__main__':print(json.dumps(audit(sys.argv[1]),indent=2))
