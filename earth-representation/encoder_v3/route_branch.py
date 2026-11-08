"""Initialize either planned route from the exact complete shared checkpoint."""
import random
import numpy as np
import torch
from .gates import require_jepa,require_training
from .participation_encoder import ParticipationEncoder
from .pilot_protocol import optimizer_groups
from .route_control import checkpoint_sha,frozen_identity,inspect_route_checkpoint,restore_route_checkpoint,_sha,_check_modules
from .route_step import RouteSchedule
from .training import JEPAPredictor,make_teacher


def _restore_rng(rng):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
    for device,state in (rng['cuda'] or {}).items():torch.cuda.set_rng_state(state,device=int(device))


def initialize_route_branch(*,config,prefix_checkpoint,frozen_ids,phase,encoder_factory,decoder_factory,ab_stable,predictor_factory=None,next_AB_input_signature=None,next_input_signature=None):
    require_training(config)
    if phase not in ('A+B','A+B+JEPA'):raise ValueError('Only the two planned branches are supported')
    if frozen_ids.get('formal_config_sha256')!=frozen_identity(config):raise ValueError('Frozen formal configuration differs')
    settings=config.get('formal_step')
    if not isinstance(settings,dict):raise ValueError('Frozen formal_step configuration required')
    latent=phase=='A+B+JEPA'
    if latent:
        require_jepa(config)
        if not callable(predictor_factory):raise ValueError('Explicit reviewed Predictor factory required')
        _sha(next_AB_input_signature);_sha(next_input_signature)
    elif predictor_factory is not None or next_AB_input_signature is not None or next_input_signature is not None:
        raise ValueError('A+B inherits the exact prefix next-input signature without additional modules')
    if not callable(encoder_factory) or not callable(decoder_factory):raise ValueError('Explicit model factories required')
    digest=checkpoint_sha(prefix_checkpoint)
    state,shared=inspect_route_checkpoint(prefix_checkpoint,frozen_ids=frozen_ids,phase='shared',prefix_checkpoint_sha256=None)
    progress=shared.fork(digest,ab_stable=ab_stable)[phase]
    prefix_signature=state['sampler']['next_input_signature']
    if latent and next_AB_input_signature!=prefix_signature:raise ValueError('JEPA branch must preserve the exact next A/B batch')
    modules=dict(encoder=encoder_factory(),decoder=decoder_factory())
    encoder=modules['encoder']
    if not isinstance(encoder,ParticipationEncoder) or encoder.merge_recipe!=frozen_ids.get('merge_recipe'):raise ValueError('Reviewed Participation configuration required')
    optimizer=torch.optim.AdamW(optimizer_groups(modules.values(),settings['weight_decay']),lr=settings['peak_lr'],betas=tuple(settings['betas']),eps=settings['eps'])
    scheduler=RouteSchedule(optimizer,peak_lr=settings['peak_lr'],threshold=settings['plateau_threshold'])
    restored,_=restore_route_checkpoint(prefix_checkpoint,modules=modules,optimizer=optimizer,scheduler=scheduler,frozen_ids=frozen_ids,phase='shared',prefix_checkpoint_sha256=None)
    if checkpoint_sha(prefix_checkpoint)!=digest:raise ValueError('Prefix changed during branch initialization')
    devices={p.device for m in modules.values() for p in m.parameters()}
    if len(devices)!=1:raise ValueError('Explicit single-device branch placement required')
    if scheduler.last_update!=progress.logical_update-1 or scheduler.last_validation!=(progress.logical_update//scheduler.validation_interval)*scheduler.validation_interval:raise ValueError('Prefix scheduler is incomplete')
    # Adding new modules must not consume the next A/B training random stream.
    try:
        for module in modules.values():module.train()
        if latent:
            teacher=make_teacher(encoder);predictor=predictor_factory()
            if type(predictor) is not JEPAPredictor:raise ValueError('Only the existing reviewed Predictor architecture is supported')
            if {p.device for p in predictor.parameters()}!=devices:raise ValueError('Predictor placement differs; no implicit device migration')
            rates={g['lr'] for g in optimizer.param_groups}
            if len(rates)!=1:raise ValueError('Common frozen A/B learning rate required')
            template={k:v for k,v in optimizer.param_groups[0].items() if k!='params'}
            for group in optimizer_groups([predictor],settings['weight_decay']):optimizer.add_param_group(dict(template,**group))
            modules.update(teacher=teacher,predictor=predictor)
        _check_modules(modules,optimizer,phase)
    finally:_restore_rng(restored['rng'])
    sampler={k:v for k,v in restored['sampler'].items() if k not in ('_route_progress','_route_optimizer_names','_route_last_metrics')}
    sampler['next_AB_input_signature']=prefix_signature
    if latent:sampler['next_input_signature']=next_input_signature
    return dict(progress=progress,modules=modules,optimizer=optimizer,scheduler=scheduler,sampler=sampler,prefix_checkpoint_sha256=digest)
