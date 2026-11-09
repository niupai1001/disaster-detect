"""Review-gated, budgeted real GPU update and checkpoint-resume measurements."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import uuid
import subprocess
import sys
import torch
from .budget import PilotBudget
from .data import NativeOpticalDataset,sha256
from .joint import accumulate_joint
from .model import Encoder,ReflectanceDecoder
from .pilot_protocol import load_study,require_update_review,prepare_update,optimizer_groups,LinearWarmup
from .training import CAdamW,save_checkpoint,restore_checkpoint


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def difference(first,second):
    tensors=[];scalars=[];dtypes=[]
    def visit(a,b,path):
        if isinstance(a,torch.Tensor):
            if not isinstance(b,torch.Tensor) or a.shape!=b.shape:raise ValueError('Tensor layout differs: '+path)
            if a.dtype!=b.dtype:dtypes.append(path)
            tensors.append((a.detach().cpu().double().flatten(),b.detach().cpu().double().flatten()))
        elif isinstance(a,dict):
            if not isinstance(b,dict) or a.keys()!=b.keys():raise ValueError('State keys differ: '+path)
            for key in a:visit(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(list,tuple)):
            if type(a)!=type(b) or len(a)!=len(b):raise ValueError('Sequence differs: '+path)
            for i,(x,y) in enumerate(zip(a,b)):visit(x,y,path+'/'+str(i))
        elif type(a)!=type(b) or a!=b:scalars.append(path)
    visit(first,second,'root')
    if not tensors:return dict(max_absolute=0.,relative_l2=0.,scalar_mismatches=scalars,dtype_mismatches=dtypes,bitwise_equal=not scalars and not dtypes)
    a=torch.cat([x for x,_ in tensors]);b=torch.cat([y for _,y in tensors])
    if not bool(torch.isfinite(a).all()&torch.isfinite(b).all()):raise FloatingPointError('Nonfinite comparison state')
    return dict(max_absolute=float((a-b).abs().max()),relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-24)),scalar_mismatches=scalars,dtype_mismatches=dtypes,bitwise_equal=bool(torch.equal(a,b)) and not scalars and not dtypes)


def run(args,study,review,config):
    root=args.run_root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=root/'pilot_run.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.fsync(descriptor)
    try:
        budget=PilotBudget(root/'budget.json',study_sha256=study['candidate_sha256'])
        summary=budget.summary()
        if summary['incomplete_reservations']:raise ValueError('Uncertain prior update reservations; no blind replay')
        if summary['reserved']:raise ValueError('Existing pilot reservations; reconcile prior check rather than starting a fresh check')
        if summary['remaining_total']<8 or any(v<4 for v in summary['remaining_by_candidate'].values()):raise ValueError('Insufficient original pilot budget for matched check')
        if (root/'update_check_completed.json').exists():raise ValueError('Bounded update check already complete; no automatic repeat')
        directory=root/('update_check_'+uuid.uuid4().hex[:12]);directory.mkdir()
        started=time.perf_counter();device=review['device']
        frozen=dict(study=study['candidate_sha256'],sources=study['source_identities'],execution_review=review,code={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
        report=dict(status='RUNNING',purpose='measure_cell_readout_update_and_resume_only',frozen_ids=frozen,check_steps=[],comparisons={},formal_training_started=False,EMA='not_enabled_in_AB_check',limitations=['Real update checks consume 4 steps per optimizer from the existing 400/200 caps','No formal tolerance acceptance or optimizer superiority inferred','No complete 200-step fitting trajectories, validation convergence, BF16 or multi-GPU acceptance','Remaining budget after 8 successful check steps is at most 196 fitting updates per candidate; no automatic extension'])
        output=directory/'report.json'
        def persist():output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        def limits():
            if time.perf_counter()-started>review['max_wall_seconds']:raise RuntimeError('Reviewed wall-time cap reached')
            if torch.cuda.max_memory_allocated(device)>review['max_allocated_bytes']:raise RuntimeError('Reviewed allocated-memory cap reached')
            if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>review['max_output_bytes']:raise RuntimeError('Reviewed output-storage cap reached')
        persist();torch.set_num_threads(2);torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        dataset=NativeOpticalDataset(study['release'],study['contracts'],role='TRAIN')
        prepared=[prepare_update(study,dataset,index,device) for index in (0,1)]
        report['update_input_signatures']=[signature for _,signature in prepared];persist()
        parameters=config['candidate']
        def create(name,cell_indexed):
            torch.manual_seed(study['seed'])
            encoder=Encoder(hyper_hidden=64,local_batch_size=64,native_mode='grouped',layout_mode='indexed').to(device).eval()
            decoder=ReflectanceDecoder(indexed=True,cell_indexed=cell_indexed).to(device).eval()
            optimizer=(torch.optim.AdamW if name=='AdamW' else CAdamW)(optimizer_groups([encoder,decoder],parameters['weight_decay']),lr=parameters['peak_lr'],betas=tuple(parameters['betas']),eps=parameters['eps'])
            schedule=LinearWarmup(optimizer,peak=parameters['peak_lr'],warmup=parameters['warmup_updates'])
            return encoder,decoder,optimizer,schedule
        def step(name,branch,objects,index):
            encoder,decoder,optimizer,schedule=objects;views,signature=prepared[index];schedule.apply(index)
            encoder.zero_grad(set_to_none=True);decoder.zero_grad(set_to_none=True)
            predictions=[];supports=[]
            def capture(module,inputs,result):predictions.append(result[0].detach().cpu());supports.append(result[1].detach().cpu())
            hook=decoder.register_forward_hook(capture)
            try:result=accumulate_joint(encoder,decoder,views,logical_update=index,chunk_pixels=512,precision='fp32',clip_norm=parameters['clip_norm'])
            finally:hook.remove()
            gradients=cpu_tree({prefix+'.'+key:p.grad for prefix,module in [('encoder',encoder),('decoder',decoder)] for key,p in module.named_parameters()})
            if any(value is None for value in gradients.values()):raise RuntimeError('Missing full-model diagnostic gradient')
            limits();checkpoint=directory/(name+'_'+branch+'.pt')
            if checkpoint.exists():raise ValueError('Refusing to overwrite a checkpoint')
            token=budget.reserve(name,index)
            report['check_steps'].append(dict(optimizer=name,branch=branch,logical_input=index,reservation=token,status='RESERVED',signature=signature));persist()
            optimizer.step();torch.cuda.synchronize(device)
            sampler=dict(next_update=index+1,next_input_signature=None if index else prepared[1][1],reservation=token,branch=branch)
            save_checkpoint(checkpoint,modules={'encoder':encoder,'decoder':decoder},optimizer=optimizer,scheduler=schedule,logical_update=index+1,sampler=sampler,frozen_ids=frozen)
            budget.complete(token)
            report['check_steps'][-1].update(status='COMPLETED',checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),result=result,lr=[g['lr'] for g in optimizer.param_groups]);persist();limits()
            support=torch.cat(supports);prediction=torch.cat(predictions)
            return dict(modules=cpu_tree(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict())),optimizer=cpu_tree(optimizer.state_dict()),schedule=cpu_tree(schedule.state_dict()),prediction=prediction[support],support=support,gradients=gradients,checkpoint=checkpoint,result=result)
        try:
            for name in ('AdamW','CAdamW'):
                reference=create(name,False);baseline=step(name,'reference_after_1',reference,0);del reference;gc.collect()
                candidate=create(name,True);first=step(name,'candidate_after_1',candidate,0)
                if not torch.equal(baseline['support'],first['support']):raise ValueError('Readout optimization changed supervision support')
                report['comparisons'][name]=dict(reference_vs_cell=dict(parameters=difference(baseline['modules'],first['modules']),optimizer_state=difference(baseline['optimizer'],first['optimizer']),gradients=difference(baseline['gradients'],first['gradients']),predictions=difference(baseline['prediction'],first['prediction']),weighted_loss_absolute=abs(baseline['result']['weighted_loss']-first['result']['weighted_loss'])))
                uninterrupted=step(name,'uninterrupted_after_2',candidate,1);del candidate;gc.collect()
                resumed=create(name,True)
                encoder,decoder,optimizer,schedule=resumed
                restored=restore_checkpoint(first['checkpoint'],modules={'encoder':encoder,'decoder':decoder},optimizer=optimizer,scheduler=schedule,frozen_ids=frozen)
                if restored['logical_update']!=1 or restored['sampler']['next_update']!=1 or restored['sampler']['next_input_signature']!=prepared[1][1]:raise ValueError('Resume next-update exposure changed')
                recovered=step(name,'restored_after_2',resumed,1)
                report['comparisons'][name]['uninterrupted_vs_restored']=dict(parameters=difference(uninterrupted['modules'],recovered['modules']),optimizer_state=difference(uninterrupted['optimizer'],recovered['optimizer']),predictions=difference(uninterrupted['prediction'],recovered['prediction']),schedule_equal=uninterrupted['schedule']==recovered['schedule'],next_input_signature_equal=True)
                del resumed,encoder,decoder,optimizer,schedule,baseline,first,uninterrupted,recovered;gc.collect();persist()
            report.update(status='BUDGETED_GPU_UPDATE_AND_RESUME_MEASUREMENTS_NOT_ACCEPTANCE',budget=budget.summary(),seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(device));persist()
            marker=root/'update_check_completed.json'
            with marker.open('x',encoding='utf-8') as stream:json.dump(dict(report=str(output),sha256=sha256(output),budget=budget.summary()),stream,indent=2)
        except Exception as exc:
            report.update(status='FAILED_NO_AUTOMATIC_RETRY',error=type(exc).__name__+': '+str(exc),budget=budget.summary(),seconds=time.perf_counter()-started);persist();raise
        return output
    finally:os.close(descriptor);lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));parser.add_argument('--run-root',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--check-only',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);args=parser.parse_args()
    study=load_study(args.study);config=json.loads(args.config.read_text(encoding='utf-8'))
    try:review=require_update_review(config,study,Path(__file__).parent)
    except ValueError as exc:print(str(exc));raise SystemExit(2)
    if args.check_only:print('REVIEW_BOUND_NO_UPDATES',json.dumps(review));raise SystemExit(0)
    if not args.worker:
        args.run_root.mkdir(parents=True,exist_ok=True)
        controller=args.run_root/('controller_'+uuid.uuid4().hex[:12]+'.json')
        child=subprocess.Popen([sys.executable,'-m','encoder_v3.gpu_update_check',*sys.argv[1:],'--worker'],creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        record=dict(status='RUNNING',owned_worker_pid=child.pid,maximum_wall_seconds=review['max_wall_seconds'])
        controller.write_text(json.dumps(record,indent=2),encoding='utf-8')
        try:code=child.wait(timeout=review['max_wall_seconds'])
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            code=1;record.update(status='DEADLINE_TERMINATED_OWN_WORKER',recovery='No automatic replay; retain locks and budget reservations for audit')
        record.update(status=record['status'] if code else 'WORKER_COMPLETED',worker_exit_code=child.returncode)
        if code and record['status']=='RUNNING':record['status']='WORKER_FAILED_NO_AUTOMATIC_RETRY'
        controller.write_text(json.dumps(record,indent=2),encoding='utf-8');raise SystemExit(code)
    print(run(args,study,review,config))
