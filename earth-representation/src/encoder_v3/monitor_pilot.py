"""Bounded read-only device sampling tied to an existing pilot controller."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',required=True);p.add_argument('--expected-uuid',required=True);args=p.parse_args()
    start=time.monotonic()
    with args.output.open('x',encoding='utf-8') as out:
        while time.monotonic()-start<14400:
            try:controller=json.loads(args.controller.read_text(encoding='utf-8'))
            except json.JSONDecodeError:time.sleep(1);continue
            sample=dict(time=time.time(),controller_status=controller['status'],owned_worker_pid=controller['owned_worker_pid'],gpu_selector=args.gpu,scope='whole device including desktop/other contexts; not process-specific')
            try:
                result=subprocess.run(['nvidia-smi','--id='+args.gpu,'--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                sample.update(exit_code=result.returncode,raw=result.stdout.strip())
                if result.returncode==0 and not result.stdout.strip().startswith(args.expected_uuid+','):sample['status']='WRONG_DEVICE_EXCLUDE_SAMPLE'
            except subprocess.TimeoutExpired:sample['status']='DEVICE_QUERY_TIMEOUT'
            out.write(json.dumps(sample)+'\n');out.flush()
            if controller['status']!='RUNNING':break
            time.sleep(5)
