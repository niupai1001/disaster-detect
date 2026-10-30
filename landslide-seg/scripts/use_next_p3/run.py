import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import yaml
import json
from use_next_p2 import run as engine
from use_next_p3.protocol import audit
from use_next_p2.protocol import sha
from landslide_seg.models.use_next_p3.tasks import LatentTask
from landslide_seg.models.use_next_p3.readout import GuardedReadout

# Process-local consumer factory substitution; the immutable P2 module on disk is untouched.
engine.ScaleReadout=GuardedReadout


def check_source(config,values,release):
    """A P5 head consumes its own accepted, complete P3 endpoint."""
    source=ROOT/values['frozen_source']
    saved=engine.torch.load(source,map_location='cpu',weights_only=False)
    summary=json.loads((source.parent/'summary.json').read_text(encoding='utf8'))
    identity=saved['identity']
    pretrain=ROOT/('configs/use_next_p3/'+values['source_objective']+'_pretrain.yaml')
    if not (saved['step']==summary['updates']==10000 and summary['complete']
            and summary['identity']==identity and not identity['engineering']
            and identity['release_sha256']==sha(Path(release)/'manifest.json')
            and identity['config_sha256']==sha(pretrain)):
        raise ValueError('P5 source must be its own complete release-bound P3 endpoint')
    local={k[len('use.local.'):]:v for k,v in saved['model'].items() if k.startswith('use.local.')}
    if engine.tensor_digest(local)!=summary['final_local_sha256']:
        raise ValueError('P3 endpoint local tensor hash differs from complete summary')


if __name__=='__main__':
    config=sys.argv[sys.argv.index('--config')+1]
    values=yaml.safe_load(Path(config).read_text())['use_next_p3']
    if '--engineering' not in sys.argv and values['task']=='head':
        if '--source' in sys.argv:raise ValueError('Formal P5 must use configured own endpoint')
        check_source(config,values,sys.argv[sys.argv.index('--release')+1])
    if '--engineering' not in sys.argv and values['task']=='head' and values['source_objective']!='joint':
        marker_argument='--marker' if '--marker' in sys.argv else '--ready-p2'
        marker=json.loads(Path(sys.argv[sys.argv.index(marker_argument)+1]).read_text())
        decision=ROOT/'outputs/use_next_v2/audit/P4_DECISION.json'
        joint=ROOT/'outputs/use_next_v2/audit/P5_JOINT_HEAD_ACCEPTANCE.json'
        if json.loads(decision.read_text()).get('status') not in ('P4_SKIPPED','P4_ACCEPTED') or sha(decision)!=marker.get('p4_decision_sha256'):
            raise ValueError('P4 decision not accepted/bound before P5 remaining heads')
        if json.loads(joint.read_text()).get('status')!='P5_JOINT_HEAD_ACCEPTED' or sha(joint)!=marker.get('joint_head_acceptance_sha256'):
            raise ValueError('Fixed joint consumer prerequisite not accepted/bound')
    engine.main(task_factory=lambda root,v:LatentTask(root,v['mode'],v['objective']),phase_key='use_next_p3',audit_fn=audit,
         marker_status='READY_P3' if values['task']=='pretrain' else 'READY_P5',source_steps=10000)
