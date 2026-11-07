"""Two-route logical accounting and complete-checkpoint lineage; no training loop."""
from dataclasses import asdict,dataclass,replace
import hashlib,json,os
from pathlib import Path


def _sha(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Explicit SHA256 required')
    return value


def frozen_identity(frozen_ids):
    if not isinstance(frozen_ids,dict) or not frozen_ids:raise ValueError('Nonempty frozen identities required')
    return hashlib.sha256(json.dumps(frozen_ids,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checkpoint_sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


@dataclass(frozen=True)
class RouteProgress:
    phase: str
    logical_update: int
    run_identity: str
    prefix_update: int = 0
    prefix_checkpoint_sha256: str | None = None

    def __post_init__(self):
        _sha(self.run_identity)
        if self.phase not in ('shared','A+B','A+B+JEPA'):raise ValueError('Only the confirmed prefix and two routes are supported')
        if type(self.logical_update) is not int or not 0<=self.logical_update<=10000:raise ValueError('Route logical cap is10000 including prefix')
        if type(self.prefix_update) is not int:raise ValueError('Integer prefix cursor required')
        if self.phase=='shared':
            if self.prefix_update!=0 or self.prefix_checkpoint_sha256 is not None:raise ValueError('Shared prefix cannot refer to another prefix')
        elif not 2000<=self.prefix_update<=self.logical_update or self.prefix_update>=10000:
            raise ValueError('Branches require a completed2000-update curriculum and remaining route budget')
        else:_sha(self.prefix_checkpoint_sha256)

    @classmethod
    def begin(cls,frozen_ids):return cls('shared',0,frozen_identity(frozen_ids))

    def next(self):
        if self.logical_update==10000:raise RuntimeError('Route cap reached; savings do not extend it')
        return replace(self,logical_update=self.logical_update+1)

    def fork(self,prefix_checkpoint_sha256,*,ab_stable):
        if self.phase!='shared' or ab_stable is not True or not 2000<=self.logical_update<10000:
            raise ValueError('Fork only a stable shared prefix after the2000-update curriculum')
        _sha(prefix_checkpoint_sha256)
        return {phase:RouteProgress(phase,self.logical_update,self.run_identity,self.logical_update,prefix_checkpoint_sha256) for phase in ('A+B','A+B+JEPA')}


def committed_update_counts(routes):
    if set(routes)!= {'A+B','A+B+JEPA'}:raise ValueError('Exactly the two planned routes required')
    a,b=routes['A+B'],routes['A+B+JEPA']
    if a.phase!='A+B' or b.phase!='A+B+JEPA' or (a.run_identity,a.prefix_update,a.prefix_checkpoint_sha256)!=(b.run_identity,b.prefix_update,b.prefix_checkpoint_sha256):
        raise ValueError('Routes must descend from the exact same complete checkpoint')
    return dict(logical_by_route={k:v.logical_update for k,v in routes.items()},shared_committed_updates=a.prefix_update,unique_committed_updates=a.logical_update+b.logical_update-a.prefix_update)


def _module_names(phase):
    return {'encoder','decoder','teacher','predictor'} if phase=='A+B+JEPA' else {'encoder','decoder'}


def _check_modules(modules,optimizer,phase):
    import torch
    if type(optimizer) is not torch.optim.AdamW:raise ValueError('Only the user-selected AdamW is supported')
    if set(modules)!=_module_names(phase):raise ValueError('Complete route module set required')
    all_parameters=[p for module in modules.values() for p in module.parameters()]
    if len({id(p) for p in all_parameters})!=len(all_parameters):raise ValueError('Teacher and trainable module parameters must be independent')
    if phase=='A+B+JEPA':
        teacher=modules['teacher']
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):raise ValueError('Teacher must be eval and stopped-gradient')
    expected=[p for name,m in modules.items() if name!='teacher' for p in m.parameters() if p.requires_grad]
    if len({id(p) for p in expected})!=len(expected):raise ValueError('Independent route modules required')
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if len({id(p) for p in actual})!=len(actual) or {id(p) for p in actual}!={id(p) for p in expected}:
        raise ValueError('Optimizer must contain exactly the route trainable parameters')



def _optimizer_names(modules,optimizer):
    names={id(p):name+'.'+key for name,module in modules.items() if name!='teacher' for key,p in module.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]

def save_route_checkpoint(path,*,progress,modules,optimizer,scheduler,sampler,frozen_ids):
    from .training import save_checkpoint
    if progress.run_identity!=frozen_identity(frozen_ids):raise ValueError('Frozen run identity differs')
    if type(sampler.get('next_update')) is not int or sampler['next_update']!=progress.logical_update:raise ValueError('Sampler and completed logical cursor differ')
    _sha(sampler.get('next_input_signature'))
    if {'_route_progress','_route_optimizer_names'} & sampler.keys():raise ValueError('Route checkpoint metadata cannot be overwritten')
    _check_modules(modules,optimizer,progress.phase)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix(path.suffix+'.route.lock')
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(descriptor,str(os.getpid()).encode());os.fsync(descriptor)
        if path.exists() or path.with_suffix(path.suffix+'.tmp').exists():raise FileExistsError('Preserve complete and interrupted checkpoint artifacts')
        save_checkpoint(path,modules=modules,optimizer=optimizer,scheduler=scheduler,logical_update=progress.logical_update,sampler=dict(sampler,_route_progress=asdict(progress),_route_optimizer_names=_optimizer_names(modules,optimizer)),frozen_ids=frozen_ids)
        return checkpoint_sha(path)
    finally:
        os.close(descriptor);lock.unlink()


def inspect_route_checkpoint(path,*,frozen_ids,phase,prefix_checkpoint_sha256):
    import torch
    before=checkpoint_sha(path);state=torch.load(path,map_location='cpu',weights_only=False)
    required={'modules','optimizer','scheduler','logical_update','sampler','frozen_ids','rng'}
    if not isinstance(state,dict) or not required<=state.keys() or state['frozen_ids']!=frozen_ids:raise ValueError('Complete checkpoint and exact frozen identities required')
    try:progress=RouteProgress(**state['sampler']['_route_progress'])
    except (KeyError,TypeError) as exc:raise ValueError('Route progress missing or malformed') from exc
    if (progress.phase,progress.prefix_checkpoint_sha256,progress.run_identity)!=(phase,prefix_checkpoint_sha256,frozen_identity(frozen_ids)):raise ValueError('Route or common-prefix identity differs')
    if type(state['logical_update']) is not int or type(state['sampler']['next_update']) is not int or state['logical_update']!=progress.logical_update or state['sampler']['next_update']!=progress.logical_update:raise ValueError('Checkpoint/sampler/progress cursor mismatch')
    _sha(state['sampler'].get('next_input_signature'))
    if not isinstance(state['modules'],dict) or set(state['modules'])!=_module_names(phase) or not isinstance(state['optimizer'],dict) or not {'state','param_groups'}<=state['optimizer'].keys() or not isinstance(state['scheduler'],dict) or not state['scheduler']:raise ValueError('Incomplete model/optimizer/scheduler state')
    names=state['sampler'].get('_route_optimizer_names')
    if not isinstance(names,list) or len(names)!=len(state['optimizer']['param_groups']) or any(not isinstance(group,list) or any(not isinstance(n,str) for n in group) or len(group)!=len(saved['params']) for group,saved in zip(names,state['optimizer']['param_groups'])):raise ValueError('Optimizer parameter order metadata required')
    if not isinstance(state['rng'],dict) or not {'python','numpy','torch','cuda'}<=state['rng'].keys():raise ValueError('Complete random state required')
    if checkpoint_sha(path)!=before:raise ValueError('Checkpoint changed during inspection')
    return state,progress


def fork_route_checkpoint(path,*,frozen_ids,ab_stable):
    before=checkpoint_sha(path)
    _,progress=inspect_route_checkpoint(path,frozen_ids=frozen_ids,phase='shared',prefix_checkpoint_sha256=None)
    if checkpoint_sha(path)!=before:raise ValueError('Prefix checkpoint changed before fork')
    return progress.fork(before,ab_stable=ab_stable)


def restore_route_checkpoint(path,*,modules,optimizer,scheduler,frozen_ids,phase,prefix_checkpoint_sha256):
    from .training import restore_checkpoint
    before=checkpoint_sha(path)
    inspected,progress=inspect_route_checkpoint(path,frozen_ids=frozen_ids,phase=phase,prefix_checkpoint_sha256=prefix_checkpoint_sha256)
    _check_modules(modules,optimizer,phase)
    if _optimizer_names(modules,optimizer)!=inspected['sampler']['_route_optimizer_names']:raise ValueError('Optimizer parameter order differs; no state was restored')
    state=restore_checkpoint(path,modules=modules,optimizer=optimizer,scheduler=scheduler,frozen_ids=frozen_ids)
    if checkpoint_sha(path)!=before:raise ValueError('Checkpoint changed during restore')
    return state,progress
