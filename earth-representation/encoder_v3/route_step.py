"""One gated Participation update and durable checkpoint; no automatic run loop."""
import hashlib,json,math,os,time
from pathlib import Path
import torch
from .adaptive_joint import accumulate_adaptive_joint
from .gates import require_jepa,require_training
from .joint import layout
from .pair_geometry import reviewed_pair_task,frame_for_task
from .participation_encoder import ParticipationEncoder
from .pilot_protocol import LinearWarmup,optimizer_groups
from .route_control import frozen_identity,_sha,_check_modules,save_route_checkpoint
from .streaming import task_signature
from .training import CheckedPlateau,update_teacher


class RouteSchedule:
    def __init__(self,optimizer,*,peak_lr,threshold,warmup_updates=100,validation_interval=250,max_updates=10000):
        if isinstance(peak_lr,bool) or isinstance(threshold,bool) or not math.isfinite(threshold) or threshold<0:raise ValueError('Explicit finite learning rate and improvement threshold required')
        if any(type(v) is not int or v<1 for v in (warmup_updates,validation_interval,max_updates)) or max_updates>10000:raise ValueError('Positive explicit scheduler controls required')
        self.warmup_updates=warmup_updates;self.validation_interval=validation_interval;self.max_updates=max_updates
        self.warmup=LinearWarmup(optimizer,peak=peak_lr,warmup=warmup_updates)
        self.plateau=CheckedPlateau(optimizer,threshold=threshold,patience=4,factor=.5,minimum=1e-5)
        self.last_update=-1;self.last_validation=0
    def before_update(self,index):
        if type(index) is not int or index!=self.last_update+1 or not 0<=index<self.max_updates:raise ValueError('Scheduler cursor must advance within the frozen route cursor')
        if self.last_validation!=(index//self.validation_interval)*self.validation_interval:raise ValueError('Previous fixed validation is missing')
        if index<self.warmup_updates:self.warmup.apply(index)
        self.last_update=index
    def validate(self,metric,*,completed_update):
        if completed_update!=self.last_update+1 or completed_update%self.validation_interval or completed_update<=self.last_validation:raise ValueError('Only the next fixed validation is accepted')
        result=self.plateau.step(metric);self.last_validation=completed_update;return result
    def state_dict(self):
        return dict(kind='route_schedule_v2',warmup_updates=self.warmup_updates,validation_interval=self.validation_interval,max_updates=self.max_updates,warmup=self.warmup.state_dict(),plateau=self.plateau.state_dict(),last_update=self.last_update,last_validation=self.last_validation)
    def load_state_dict(self,state):
        if state.get('kind') not in ('route_schedule_v1','route_schedule_v2'):raise ValueError('Frozen scheduler configuration differs')
        warmup_updates=state.get('warmup_updates',100);validation_interval=state.get('validation_interval',250);max_updates=state.get('max_updates',10000)
        if (warmup_updates,validation_interval,max_updates)!=(self.warmup_updates,self.validation_interval,self.max_updates) or any(state['plateau'][k]!=getattr(self.plateau,k) for k in ('threshold','patience','factor','minimum')):raise ValueError('Frozen scheduler configuration differs')
        index,validation=state['last_update'],state['last_validation']
        if type(index) is not int or not -1<=index<self.max_updates or type(validation) is not int or validation<0 or validation%self.validation_interval or validation>index+1:raise ValueError('Malformed scheduler progress')
        if state['warmup']['last_update']!=min(index,self.warmup_updates-1):raise ValueError('Warmup progress differs from logical cursor')
        self.warmup.load_state_dict(state['warmup']);self.plateau.load_state_dict(state['plateau']);self.last_update=index;self.last_validation=validation



def validate_frozen_task_source(task,frozen_ids):
    expected=frozen_ids.get('source_identities');actual=task.metadata.get('frozen_source_ids')
    if task.metadata.get('role')!='TRAIN' or not isinstance(expected,dict) or not expected or not isinstance(actual,dict):raise ValueError('Only matching frozen TRAIN observations are accepted')
    frame=frame_for_task(task)
    native=dict(actual)
    if frame is not None:
        allowed=frozen_ids.get('pair_geometry_records')
        if not isinstance(allowed,list) or not allowed:raise ValueError('Frozen run has no approved pair geometry records')
        for value in allowed:_sha(value)
        if frame.record_sha256 not in allowed:raise ValueError('Pair geometry is reviewed but not frozen for this run')
        if native.pop('pair_geometry_record_sha256',None)!=frame.record_sha256:raise ValueError('Pair geometry identity differs')
    if native!=expected:raise ValueError('Native source identity differs or has unknown extensions')

def update_signature(views,jepa_views=()):
    rows=[]
    for view in views:
        plan=view.plan
        rows.append(dict(task=task_signature(view.task),loss_source=view.loss_source,train_stds=view.train_stds,origin=None if plan is None else plan.origin_xy,physical_patch_m=None if plan is None else plan.physical_patch_m,target_indices=[] if plan is None else [dict(band=b.target.band.record_id,indices=hashlib.sha256(b.indices.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),supported=b.supported_count) for b in plan.bands]))
        if view.absence_reason is not None:rows[-1]['absence_reason']=view.absence_reason
    latent=[dict(task=task_signature(reviewed_pair_task(v.source,v.target,v.admission)),source_kwargs=v.source_encoder_kwargs,target_kwargs=v.target_encoder_kwargs,loss_source=v.loss_source) for v in jepa_views]
    return frozen_identity(dict(AB=rows,JEPA=latent))


def _sync(modules):
    for device in sorted({p.device.index for m in modules.values() for p in m.parameters() if p.device.type=='cuda'}):torch.cuda.synchronize(device)


def _json(path,payload,*,new=False):
    if new:
        with path.open('x',encoding='utf-8') as stream:json.dump(payload,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    else:
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('w',encoding='utf-8') as stream:json.dump(payload,stream,ensure_ascii=False,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        os.replace(temp,path)


def execute_route_update(*,config,progress,frozen_ids,modules,optimizer,scheduler,views,expected_input_signature,sampler_after,directory,jepa_views=(),jepa_weight=None,latent_loss_fn=None,ab_stable=False,validation=None,expected_AB_input_signature=None):
    started=time.perf_counter()
    require_training(config)
    if frozen_ids.get('formal_config_sha256')!=frozen_identity(config) or progress.run_identity!=frozen_identity(frozen_ids):raise ValueError('Frozen formal configuration or route identity differs')
    after=progress.next()  # Refuse the next optimizer operation before exceeding the cap.
    settings=config.get('formal_step')
    if not isinstance(settings,dict):raise ValueError('Explicit frozen formal_step configuration required')
    required={'precision','chunk_pixels','microbatch_observations','clip_norm','peak_lr','plateau_threshold','weight_decay','betas','eps','ema_coefficient'}
    if set(settings)!=required:raise ValueError('Complete formal_step settings required; no candidate defaults')
    if settings['precision']!='fp32' or type(settings['chunk_pixels']) is not int or settings['chunk_pixels']<1 or type(settings['microbatch_observations']) is not int or settings['microbatch_observations'] not in (1,2,4,8):raise ValueError('Only explicit FP32 observation execution is supported')
    if isinstance(settings['clip_norm'],bool) or not math.isfinite(settings['clip_norm']) or settings['clip_norm']<=0:raise ValueError('Positive reviewed clipping required')
    if type(scheduler) is not RouteSchedule or scheduler.warmup.optimizer is not optimizer or scheduler.plateau.optimizer is not optimizer or scheduler.warmup.peak!=settings['peak_lr'] or scheduler.plateau.threshold!=settings['plateau_threshold'] or scheduler.last_update!=progress.logical_update-1:raise ValueError('Complete scheduler and logical progress must match')
    _check_modules(modules,optimizer,progress.phase)
    if any(p.is_floating_point() and p.dtype!=torch.float32 for m in modules.values() for p in m.parameters()):raise ValueError('FP32 execution requires FP32 learned parameters')
    encoder,decoder=modules['encoder'],modules['decoder']
    if not isinstance(encoder,ParticipationEncoder) or encoder.merge_recipe!=frozen_ids.get('merge_recipe'):raise ValueError('Reviewed Participation Encoder configuration required')
    for view in views:validate_frozen_task_source(view.task,frozen_ids)
    layout(views)
    expected_decay={id(p):group['weight_decay'] for group in optimizer_groups([m for k,m in modules.items() if k!='teacher'],settings['weight_decay']) for p in group['params']}
    for group in optimizer.param_groups:
        if tuple(group['betas'])!=tuple(settings['betas']) or group['eps']!=settings['eps'] or group.get('maximize',False) or group.get('amsgrad',False) or any(group['weight_decay']!=expected_decay[id(p)] for p in group['params']):raise ValueError('Optimizer or bias/LayerNorm decay differs from frozen configuration')
    latent=progress.phase=='A+B+JEPA'
    if latent:
        require_jepa(config)
        if ab_stable is not True or progress.logical_update<2000 or not callable(latent_loss_fn) or jepa_weight is None or not math.isfinite(jepa_weight) or jepa_weight<=0 or not 0<=settings['ema_coefficient']<1:raise ValueError('Explicit reviewed JEPA conditions required')
    elif jepa_views or jepa_weight is not None or latent_loss_fn is not None:raise ValueError('JEPA cannot enter the shared or A+B route')
    for view in jepa_views:
        if view.target.metadata.get('role')!='TRAIN' or view.target.metadata.get('frozen_source_ids')!=frozen_ids.get('source_identities'):raise ValueError('Teacher target source differs from frozen TRAIN data')
        validate_frozen_task_source(reviewed_pair_task(view.source,view.target,view.admission),frozen_ids)
    _sha(expected_input_signature)
    signature=update_signature(views,jepa_views)
    AB_signature=update_signature(views)
    if latent:
        _sha(expected_AB_input_signature);_sha(sampler_after.get('next_AB_input_signature'))
        if AB_signature!=expected_AB_input_signature:raise ValueError('A/B branch exposure differs from the common-prefix stream')
    elif expected_AB_input_signature is not None and AB_signature!=expected_AB_input_signature:raise ValueError('A/B exposure expectation differs')
    if signature!=expected_input_signature:raise ValueError('Next members/masks/targets differ from checkpoint expectation')
    if type(sampler_after.get('next_update')) is not int or sampler_after['next_update']!=after.logical_update:raise ValueError('Next sampler cursor differs')
    _sha(sampler_after.get('next_input_signature'))
    if {'_route_progress','_route_optimizer_names','_route_last_metrics'} & sampler_after.keys():raise ValueError('Reserved checkpoint metadata cannot be replaced')
    if after.logical_update%250==0 and not callable(validation):raise ValueError('Fixed validation must be supplied before the scheduled update')
    directory=Path(directory)/progress.phase;directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/f'u{after.logical_update:06d}.pt';attempt=directory/f'u{after.logical_update:06d}.attempt.json';lock=directory/'.update.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);owned_attempt=False;record=dict(status='PREPARED',pid=os.getpid(),phase=progress.phase,logical_update_before=progress.logical_update,logical_update_after=after.logical_update,run_identity=progress.run_identity,input_signature=signature,AB_input_signature=AB_signature,next_input_signature=sampler_after['next_input_signature'],checkpoint=str(checkpoint),checkpoint_sha256=None,events=[],optimizer_may_have_executed=False)
    try:
        os.write(descriptor,str(os.getpid()).encode());os.fsync(descriptor)
        if checkpoint.exists() or checkpoint.with_suffix('.pt.tmp').exists() or attempt.exists() or attempt.with_suffix('.json.tmp').exists():raise FileExistsError('Preserve previous complete/interrupted attempt; restore and reconcile before retry')
        _json(attempt,record,new=True);owned_attempt=True
        def event(name):
            record['events'].append(dict(kind=name,elapsed_wall_seconds=time.perf_counter()-started));_json(attempt,record)
        scheduler.before_update(progress.logical_update)
        result=accumulate_adaptive_joint(encoder,decoder,views,logical_update=progress.logical_update,chunk_pixels=settings['chunk_pixels'],clip_norm=settings['clip_norm'],microbatch_observations=settings['microbatch_observations'],jepa_views=jepa_views,teacher=modules.get('teacher'),predictor=modules.get('predictor'),jepa_weight=jepa_weight,latent_loss_fn=latent_loss_fn,curriculum_end_update=2000 if latent else None,ab_stable=ab_stable)
        event('FORWARD_BACKWARD_QUEUED');record['optimizer_may_have_executed']=True;event('OPTIMIZER_ABOUT_TO_EXECUTE')
        optimizer.step()
        if latent:update_teacher(modules['teacher'],encoder,settings['ema_coefficient'])
        _sync(modules);event('UPDATE_AND_EMA_COMPLETED')
        validation_result=None
        if after.logical_update%250==0:
            training={k:m.training for k,m in modules.items()}
            try:
                for m in modules.values():m.eval()
                with torch.no_grad():metric=validation(modules,after.logical_update)
            finally:
                for k,m in modules.items():m.train(training[k])
            validation_result=scheduler.validate(metric,completed_update=after.logical_update);_sync(modules);event('VALIDATION_COMPLETED')
        draft=dict(result=result,logical_update=after.logical_update,input_signature=signature,AB_input_signature=record['AB_input_signature'],lr=[g['lr'] for g in optimizer.param_groups],validation=validation_result)
        digest=save_route_checkpoint(checkpoint,progress=after,modules=modules,optimizer=optimizer,scheduler=scheduler,sampler=dict(sampler_after,_route_last_metrics=draft),frozen_ids=frozen_ids)
        record.update(status='CHECKPOINT_COMMITTED',checkpoint_sha256=digest);event('CHECKPOINT_COMMITTED')
        metrics=dict(draft,checkpoint_sha256=digest,end_to_end_wall_seconds=time.perf_counter()-started,timing_scope='Synchronized update/EMA and checkpoint; includes control checks and I/O, excludes caller data read/transfer')
        _json(directory/f'u{after.logical_update:06d}.metrics.json',metrics,new=True)
        record['status']='COMPLETE';event('METRICS_COMMITTED');return after,metrics
    except BaseException as exc:
        # Keep uncertain operations visible; never replay or extend budget here.
        if owned_attempt:
            record.update(status='FAILED_RECONCILE_REQUIRED',error=dict(type=type(exc).__name__,message=str(exc)),elapsed_wall_seconds=time.perf_counter()-started)
            try:_json(attempt,record)
            except OSError:pass
        raise
    finally:
        os.close(descriptor);lock.unlink()
