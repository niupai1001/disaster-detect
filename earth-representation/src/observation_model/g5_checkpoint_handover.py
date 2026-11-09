"""One checkpoint-boundary switch to recovery every50; no restart loop."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    root=Path(__file__).resolve().parent.parent.parent;g=root/'observation_model/outputs/encoder_v2/G5_native_bank'
    log=(g/'acceleration/checkpoint_handover.log').open('a',encoding='utf-8',buffering=1);sys.stdout=sys.stderr=log
    control=g/'acceleration/checkpoint_handover.json';config=json.loads(control.read_text(encoding='utf-8'));pid=config['worker_pid'];target=config['target_update']
    while True:
        status=json.loads((g/'status.json').read_text(encoding='utf-8'))
        if status.get('pid')!=pid or status['status'].startswith(('FAILED','COMPLETED')):
            config.update(status='CANCELLED_WORKER_CHANGED');break
        if status['optimizer_updates']>target+1:target=(status['optimizer_updates']//500+1)*500
        history=g/'evaluation_history.jsonl';last=json.loads(history.read_text(encoding='utf-8').splitlines()[-1])
        if target<=status['optimizer_updates']<=target+1 and last['update']==target and (g/'last.pt').stat().st_mtime_ns>history.stat().st_mtime_ns:
            command=f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if ($p.CommandLine -notlike '*observation_model.g5_detached_worker*') {{throw 'Worker identity changed'}}; Stop-ScheduledTask -TaskName EarthRepresentation_G5_Authorized"
            subprocess.run(['powershell.exe','-NoProfile','-Command',command],check=True)
            lock=g/'worker.lock'
            if lock.exists():assert lock.read_text().strip()==str(pid);lock.unlink()
            shutil.copy2(g/'status.json',g/'acceleration/before_checkpoint50_status.json')
            shutil.copy2(g/'last.pt',g/'acceleration/before_checkpoint50_last.pt')
            display=root/'observation_model/outputs/encoder_v2/tvlp/encoder_v2_G5_scheduled_live';archive=g/'viewer_fix/display_before_checkpoint50'
            assert not archive.exists();shutil.move(str(display),str(archive))
            cursor=g/'encoder_v2_G5_scheduled_live_bridge_state.json'
            if cursor.exists():shutil.move(str(cursor),str(g/'viewer_fix/cursor_before_checkpoint50.json'))
            subprocess.run(['powershell.exe','-NoProfile','-Command','Start-ScheduledTask -TaskName EarthRepresentation_G5_Authorized'],check=True)
            config.update(status='LAUNCHED_CHECKPOINT50_RUNTIME',checkpoint_update=target);break
        time.sleep(.5)
    control.write_text(json.dumps(config,indent=2),encoding='utf-8');print(config,flush=True)


if __name__=='__main__':main()
