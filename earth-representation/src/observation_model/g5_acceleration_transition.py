"""One guarded handover after a fully validated checkpoint; no new experiment."""
from datetime import datetime,timezone
import json
import os
import shutil
import subprocess
import sys
import time

import torch

from .g5_protocol import G5
from .expansion_data import ROOT,read,save,digest


def main():
    path=G5/'acceleration/transition.json';transition=read(path)
    assert transition['status']=='ARMED_FOR_VALIDATED_CHECKPOINT' and read(G5/'acceleration_check.json')['status']=='PASS'
    transition.update(controller_pid=os.getpid(),controller_started_utc=datetime.now(timezone.utc).isoformat());save(path,transition)
    target=transition['target_checkpoint_update'];pid=int(transition['original_worker_pid'])
    while True:
        try:status=read(G5/'status.json')
        except json.JSONDecodeError:time.sleep(1);continue
        if status.get('pid')!=pid or status['status'].startswith(('FAILED','COMPLETED')):
            transition.update(status='STOPPED_WITHOUT_HANDOVER',reason='Original worker changed or stopped; no process was terminated');save(path,transition);return
        if status['optimizer_updates']>target:
            target=((status['optimizer_updates']//500)+1)*500;transition['target_checkpoint_update']=target;save(path,transition)
        if status['optimizer_updates']!=target:time.sleep(1);continue
        history=G5/'evaluation_history.jsonl'
        history_text=history.read_text(encoding='utf-8')
        if not history_text.endswith('\n'):time.sleep(1);continue
        rows=[json.loads(line) for line in history_text.splitlines()]
        if not rows or rows[-1]['update']!=target:time.sleep(1);continue
        checkpoint=G5/'last.pt'
        # Last is written again after validation, best selection and LR/stop state.
        if checkpoint.stat().st_mtime_ns<=history.stat().st_mtime_ns:time.sleep(1);continue
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['state']['update']!=target:time.sleep(1);continue
        for relative,sha in read(G5/'acceleration/manifest.json')['sha256'].items():
            if digest(ROOT/relative)!=sha:
                transition.update(status='CANCELLED_SOURCE_CHANGED_KEEP_ORIGINAL',changed=relative);save(path,transition);return
        transition.update(status='STOPPING_AT_VALIDATED_CHECKPOINT',checkpoint_update=target,
            checkpoint_sha256=digest(checkpoint),validation_score=rows[-1]['score'],pause_started_utc=datetime.now(timezone.utc).isoformat());save(path,transition)
        # PID and command must both match the original authorized G5 worker.
        command=f"$p = Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if ($null -eq $p -or $p.CommandLine -notlike '*-m observation_model.g5_train --run*') {{ throw 'Original G5 worker identity changed' }}; Stop-Process -Id {pid} -ErrorAction Stop"
        subprocess.run(['powershell.exe','-NoProfile','-Command',command],check=True)
        lock=G5/'worker.lock'
        if lock.exists():
            assert lock.read_text().strip()==str(pid);lock.unlink()
        after=read(G5/'status.json');transition.update(completed_updates_before_restart=after['optimizer_updates'],
            newly_replayed_updates=max(0,after['optimizer_updates']-target),status='STARTING_FAST_RUNTIME')
        shutil.copy2(checkpoint,G5/'acceleration/checkpoint_before_handover.pt')
        shutil.copy2(G5/'best.pt',G5/'acceleration/best_before_handover.pt')
        save(path,transition)
        with (G5.parent/'G5_fast.log').open('a',encoding='utf-8') as log:
            worker=subprocess.Popen([sys.executable,'-X','utf8','-m','observation_model.g5_fast_train'],cwd=ROOT,stdout=log,stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
        transition.update(status='FAST_RUNTIME_LAUNCHED',fast_launcher_pid=worker.pid,launch_utc=datetime.now(timezone.utc).isoformat());save(path,transition)
        for _ in range(120):
            if worker.poll() is not None:
                transition.update(status='FAST_RUNTIME_FAILED',exit_code=worker.returncode);save(path,transition);return
            try:current=read(G5/'status.json')
            except json.JSONDecodeError:time.sleep(1);continue
            if current.get('pid')!=pid and current['optimizer_updates']>target:
                transition.update(status='FAST_RUNTIME_PROGRESS_VERIFIED',fast_worker_pid=current['pid'],
                    observed_optimizer_updates=current['optimizer_updates'],last_batch_seconds=current.get('last_batch_seconds'))
                save(path,transition);return
            time.sleep(1)
        transition.update(status='FAST_RUNTIME_PROGRESS_PENDING_CHECK');save(path,transition)


if __name__=='__main__':main()
