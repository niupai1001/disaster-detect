"""Bounded supervised USE queue; stop at first failed engineering/preflight gate."""
import json
import argparse
import subprocess
import sys
import time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/use'
NAMES=('baseline','mean','self_mean','self_query','physical','random','capacity','mean_mixed','physical_mixed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=(*NAMES,'pretrained_mixed'))
    parser.add_argument('--finish-stage',action='store_true',help='Wait for external acquisition, audit, pretrain and fine-tune the tenth group')
    parser.add_argument('--resume-completion',action='store_true',help='Resume a failed completion queue only after inspecting the error')
    parser.add_argument('--device',default='cuda:1',choices=['cuda:0','cuda:1']);args=parser.parse_args()
    if args.finish_stage:
        if args.only:parser.error('--finish-stage cannot be combined with --only')
        return finish_stage(resume=args.resume_completion)
    if args.resume_completion:parser.error('--resume-completion requires --finish-stage')
    status_path=OUT/('queue_status_'+args.only+'.json' if args.only else 'queue_status.json')
    if not (OUT/'engineering_acceptance.json').is_file():
        raise RuntimeError('Run audit_use first')
    state=json.loads(status_path.read_text()) if status_path.exists() else {'jobs':[]}
    state.update(status='running');state.pop('error',None)
    def save():
        tmp=status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=1),encoding='utf-8')
        for attempt in range(50):
            try:
                tmp.replace(status_path);break
            except PermissionError:
                if attempt==49:raise
                time.sleep(.1)
    def run(args,label):
        command=[sys.executable,'-X','utf8',*args]
        row={'label':label,'command':command,'started':time.time()};state['jobs'].append(row);save()
        with (OUT/(label+'.log')).open('a',encoding='utf-8') as log:
            rc=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row.update(exit_code=rc,ended=time.time());save()
        if rc:raise RuntimeError(f'{label} failed: see its log')
    try:
        for name in ((args.only,) if args.only else NAMES):
            path=ROOT/f'configs/segformer/use/{name}.yaml'
            formal=yaml.safe_load(path.read_text())
            run(['scripts/check_research_protocol.py','--config',str(path)],'preflight_'+name)
            smoke=json.loads(json.dumps(formal));s=smoke['segformer']
            s['experiment']='smoke_'+name
            s['train'].update(max_steps=100,stop_updates=100,report_steps=[100])
            sp=path.with_name('smoke_'+name+'.yaml')
            if not sp.exists():sp.write_text(yaml.safe_dump(smoke,sort_keys=False),encoding='utf-8')
            smoke_dir=OUT/s['data']['version']/s['experiment']/str(s['seed'])
            summary=smoke_dir/'train_summary.json'
            if not summary.is_file():
                run(['scripts/check_research_protocol.py','--config',str(sp)],'preflight_smoke_'+name)
                run(['scripts/train_segformer.py','--config',str(sp),'--device',args.device],'smoke_'+name)
            evidence=json.loads(summary.read_text())
            if not evidence.get('complete') or evidence.get('global_step')!=100:
                raise RuntimeError('Incomplete smoke')
            t=formal['segformer'];target=OUT/t['data']['version']/t['experiment']/str(t['seed'])/'train_summary.json'
            if target.exists():
                evidence=json.loads(target.read_text())
                if not evidence.get('complete') or evidence['global_step']!=2000:raise RuntimeError('Existing incomplete formal run; inspect before resume')
            else:
                run(['scripts/train_segformer.py','--config',str(path),'--device',args.device],'formal_'+name)
        state['status']='supervised_training_finished_requires_evaluation'
    except Exception as exc:
        state.update(status='failed',error=str(exc));raise
    finally:save()


def finish_stage(resume=False):
    """Continue the approved pipeline, stopping on every failed data/training gate."""
    import hashlib
    status=OUT/'completion_status.json'
    state=json.loads(status.read_text()) if resume else dict(status='waiting_external',jobs=[])
    if resume and state['status']!='failed':raise ValueError('Only a failed completion queue can be explicitly resumed')
    state.update(status='waiting_external');state.pop('error',None)
    def save():
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=1),encoding='utf-8');temporary.replace(status)
    def run(arguments,label):
        state['status']=label
        row=dict(command=[sys.executable,'-X','utf8',*arguments],started=time.time());state['jobs'].append(row);save()
        with (OUT/(label+'.log')).open('a',encoding='utf-8') as log:
            result=subprocess.run(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row.update(exit_code=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(label+' failed; inspect '+label+'.log')
    try:
        if status.exists() and not resume:raise FileExistsError('Completion queue already registered; inspect its process and evidence')
        save()
        while True:
            external=json.loads((OUT/'external/acquisition_status.json').read_text())
            if external['status']=='complete':break
            if external['status']!='running':raise RuntimeError('External acquisition stopped: '+external['status'])
            time.sleep(30)
        run(['scripts/prepare_use_external.py','--audit-full'],'external_full_audit')
        interface=json.loads((OUT/'segmentation_interface_acceptance.json').read_text())
        if interface.get('ok') is not True:raise RuntimeError('Complete segmentation interface acceptance required before pretraining')
        output=OUT/'pretraining'
        if not (output/'summary.json').exists():
            if output.exists() and any(output.iterdir()):raise RuntimeError('Inspect incomplete pretraining before explicit resume')
            run(['scripts/pretrain_use.py'],'external_pretraining')
        summary=json.loads((output/'summary.json').read_text())
        if summary.get('complete') is not True or summary['updates']!=10000:raise ValueError('Incomplete external pretraining')
        best=output/'best.pt';sha=hashlib.sha256(best.read_bytes()).hexdigest()
        if sha!=summary['best_sha256']:raise ValueError('Pretraining weight hash changed')
        cfg=yaml.safe_load((ROOT/'configs/segformer/use/physical_mixed.yaml').read_text())
        cfg['segformer']['experiment']='use_pretrained_mixed'
        cfg['segformer']['model'].update(use_pretrained=str(best.relative_to(ROOT)),use_pretrained_sha256=sha)
        path=ROOT/'configs/segformer/use/pretrained_mixed.yaml'
        if path.exists() and yaml.safe_load(path.read_text())!=cfg:raise ValueError('Existing tenth-group configuration differs')
        if not path.exists():path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        run(['scripts/audit_use.py','--transfer'],'pretraining_transfer_audit')
        run(['scripts/run_use.py','--only','pretrained_mixed'],'pretrained_finetuning')
        run(['scripts/audit_use.py','--evaluate-pretrained'],'pretrained_evaluation')
        run(['scripts/audit_use.py','--cost'],'cost_measurement')
        run(['scripts/summarize_use.py','--cases'],'use_figures')
        run(['scripts/query_research.py','--refresh-stages'],'use_stage_documents')
        run(['scripts/build_research_inventory.py','--no-figures'],'use_inventory')
        run(['scripts/audit_use.py','--artifacts'],'use_artifact_hashes')
        state['status']='training_and_evaluation_complete_requires_report_review';save()
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
