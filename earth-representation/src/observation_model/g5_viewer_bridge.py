"""Read-only TVLP display aliases for the existing Viewer; no model imports."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parent/'outputs/encoder_v2'
SOURCE=ROOT/'tvlp/encoder_v2_G5_nativebank_20k'
DEST=ROOT/'tvlp/encoder_v2_G5_live_view'
STATE=ROOT/'G5_native_bank/viewer_bridge_state.json'


def prepare():
    DEST.mkdir(exist_ok=True)
    if (DEST/'run.json').exists():return
    meta=json.loads((SOURCE/'run.json').read_text(encoding='utf-8'))
    meta['run_id']=DEST.name;meta['name']='G5训练曲线｜A/B真实损失与固定验证'
    meta['config']['display_only']=True;meta['config']['source_training_run_id']=SOURCE.name
    meta['config']['display_loss_definition']='train/loss=lambda_A*train_A/score+lambda_B*train_B/score; val/loss=(0.25*val_A/score+val_B/score)/1.25. Source values/timestamps/steps retained. No model execution.'
    meta['metric_definitions']={
        'train/loss':dict(direction='min',display_name='实际加权训练目标',unit='NMSE',description=meta['config']['display_loss_definition']),
        'val/loss':dict(direction='min',display_name='固定验证选模评分',unit='NMSE',description=meta['config']['display_loss_definition']),
        'train/A_nmse':dict(direction='min',display_name='A空间遮盖NMSE',unit='NMSE'),
        'train/B_nmse':dict(direction='min',display_name='B混合缺失NMSE',unit='NMSE'),
        'val/A_nmse':dict(direction='min',display_name='固定验证A NMSE',unit='NMSE'),
        'val/B_nmse':dict(direction='min',display_name='固定验证单波段B NMSE',unit='NMSE'),
        'state/learning_rate':dict(direction='none',display_name='实际学习率'),
    }
    (DEST/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')


def sync(state):
    for filename in ['metrics.jsonl','events.jsonl']:
        source=SOURCE/filename
        if not source.exists():continue
        offset=state.get(filename,0)
        if source.stat().st_size<offset:raise RuntimeError('Source log truncated; display cursor needs explicit review')
        with source.open('rb') as reader,(DEST/filename).open('a',encoding='utf-8') as writer:
            reader.seek(offset)
            while True:
                start=reader.tell();line=reader.readline()
                if not line or not line.endswith(b'\n'):break
                row=json.loads(line);output=None
                if state.get('session_filter') and row.get('session_id')!=state['session_filter'] and not (filename=='metrics.jsonl' and row.get('phase') in ['val_A','val_B']):
                    state[filename]=reader.tell();continue
                if filename=='events.jsonl':
                    output=row
                    if row.get('type')=='artifact':
                        source_asset=SOURCE/row['path'];target_asset=DEST/row['path']
                        if source_asset.is_file():
                            target_asset.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_asset,target_asset)
                else:
                    phase=row['phase'];metrics=row['metrics'];key=row.get('session_id','')+':'+str(row['global_step'])
                    pending=state.setdefault('pending',{}).setdefault(key,{})
                    if phase in ['train_A','train_B','val_A','val_B']:pending[phase]=metrics['score']
                    def alias(alias_phase,values):
                        return dict(timestamp=row['timestamp'],global_step=row['global_step'],session_id=row.get('session_id',''),
                            phase=alias_phase,metrics=values,event_id='display:'+filename+':'+str(start),
                            extensions=dict(source_file=str(source),source_byte_offset=start,derived_display_only=True))
                    if phase=='state':
                        if 'train_A' in pending and 'train_B' in pending:
                            a,b=pending['train_A'],pending['train_B']
                            writer.write(json.dumps(alias('train',dict(loss=metrics['lambda_A']*a+metrics['lambda_B']*b,A_nmse=a,B_nmse=b)),ensure_ascii=False)+'\n')
                            state['pending'].pop(key,None)
                        output=alias('state',metrics);output['event_id']+=':state'
                    elif phase=='val_B' and 'val_A' in pending:
                        a,b=pending['val_A'],pending['val_B'];output=alias('val',dict(loss=(.25*a+b)/1.25,A_nmse=a,B_nmse=b));state['pending'].pop(key,None)
                    elif phase=='capacity' or phase.startswith(('combo_','unseen_response_')):output=alias(phase,metrics)
                    if phase not in ['train_A','train_B','val_A','val_B','state']:state['pending'].pop(key,None)
                if output is not None:writer.write(json.dumps(output,ensure_ascii=False)+'\n')
                state[filename]=reader.tell()
            writer.flush();os.fsync(writer.fileno())
    state.update(pid=os.getpid(),last_sync_utc=datetime.now(timezone.utc).isoformat(),model_optimizer_updates=0)
    temp=STATE.with_suffix('.tmp')
    with temp.open('w',encoding='utf-8') as handle:
        handle.write(json.dumps(state,ensure_ascii=False));handle.flush();os.fsync(handle.fileno())
    temp.replace(STATE)


def main():
    global DEST,STATE
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--resume-view',action='store_true');parser.add_argument('--display-id',default='encoder_v2_G5_resumed_live');args=parser.parse_args()
    if args.resume_view:
        assert '/' not in args.display_id and '\\' not in args.display_id
        DEST=ROOT/'tvlp'/args.display_id;STATE=ROOT/'G5_native_bank'/(args.display_id+'_bridge_state.json')
    prepare();state=json.loads(STATE.read_text(encoding='utf-8')) if STATE.exists() else {}
    if args.resume_view:
        events=[json.loads(line) for line in (SOURCE/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        session=next(r['session_id'] for r in reversed(events) if r.get('type')=='lifecycle' and r.get('status') in ['started','resumed'])
        if state.get('session_filter') not in [None,session]:raise RuntimeError('Resume display belongs to another session; preserve it before rebuilding')
        state['session_filter']=session
        path=DEST/'run.json';meta=json.loads(path.read_text(encoding='utf-8'));meta['name']='G5恢复训练｜当前会话实时曲线｜总预算20k';meta['config']['resume_session_id']=session;path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    while True:
        try:sync(state)
        except PermissionError:
            if not args.watch:raise
            time.sleep(1);continue
        if not args.watch:break
        try:status=json.loads((ROOT/'G5_native_bank/status.json').read_text(encoding='utf-8'))
        except (PermissionError,json.JSONDecodeError):
            # The Windows writer can briefly lock/truncate this status snapshot.
            time.sleep(1);continue
        if status['status'].startswith('COMPLETED'):break
        time.sleep(5)


if __name__=='__main__':main()
