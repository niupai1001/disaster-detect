"""One manually launched scheduled G5 worker; no console and no restart loop."""
import ctypes
from datetime import datetime,timezone
import faulthandler
import json
import os
from pathlib import Path
import sys
import subprocess
import traceback
import copy
import time


def main():
    root=Path(__file__).resolve().parent.parent.parent
    base=root/'observation_model/outputs/encoder_v2';g=base/'G5_native_bank'
    if (g/'CLOSED_BY_USER.json').exists():
        raise RuntimeError('G5 ended by user; explicit new authorization required before launch')
    log=(base/'G5_scheduled.log').open('a',encoding='utf-8',buffering=1)
    sys.stdout=sys.stderr=log;faulthandler.enable(file=log,all_threads=True)
    record=dict(pid=os.getpid(),parent_pid=os.getppid(),console_window=int(ctypes.windll.kernel32.GetConsoleWindow()),started_utc=datetime.now(timezone.utc).isoformat(),entry='observation_model.g5_fast_train',automatic_restart=False)
    (g/'scheduled_worker.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('SCHEDULED_WORKER_START',json.dumps(record),flush=True)
    bridge=None
    try:
        assert record['console_window']==0,'Worker must not inherit a console'
        if '--probe' in sys.argv:
            time.sleep(12)
            record['status']='DETACHED_PROBE_PASS'
            return
        from . import g5_train
        context={}
        restore=g5_train.restore
        def capture_restore(path,model,optimizer,device):
            payload=restore(path,model,optimizer,device)
            context.update(model=model,optimizer=optimizer,state=payload['state'],device=device,
                wall=payload['state']['wall_seconds'],started=time.perf_counter(),saved=payload['state']['update'])
            return payload
        g5_train.restore=capture_restore
        original_run=g5_train.V2Run
        def run_with_bridge(*args,**kwargs):
            nonlocal bridge
            run=original_run(*args,**kwargs)
            heartbeat=run.heartbeat
            def heartbeat_with_recovery(step):
                heartbeat(step)
                state=context['state']
                if step==state['update'] and step>context['saved'] and step%50==0 and step%500!=0:
                    snapshot=copy.deepcopy(state)
                    snapshot['wall_seconds']=context['wall']+time.perf_counter()-context['started']
                    snapshot['recovery_checkpoint_interval']=50
                    g5_train.checkpoint(g/'last.pt',context['model'],context['optimizer'],snapshot,run,context['device'])
                    with (g/'last.pt').open('r+b') as handle:os.fsync(handle.fileno())
                    context['saved']=step
                    print('FULL_RECOVERY_CHECKPOINT',step,flush=True)
            run.heartbeat=heartbeat_with_recovery
            bridge=subprocess.Popen([sys.executable,'-X','utf8','-m','observation_model.g5_viewer_bridge','--watch','--resume-view','--display-id','encoder_v2_G5_scheduled_live'],cwd=root,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
            return run
        g5_train.V2Run=run_with_bridge
        from .g5_fast_train import main as train
        train()
    except BaseException:
        traceback.print_exc();record['status']='FAILED';raise
    else:record['status']='COMPLETED'
    finally:
        if bridge is not None and bridge.poll() is None:bridge.terminate()
        record['ended_utc']=datetime.now(timezone.utc).isoformat()
        (g/'scheduled_worker.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        log.flush()


if __name__=='__main__':main()
