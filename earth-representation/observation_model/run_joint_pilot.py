"""One bounded pilot and its fixed geographic validation; old queue stays paused."""

from .execution_gate import require_training_authorization
if __name__ == "__main__": require_training_authorization()
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'observation_model/outputs/joint_v3'


def main():
    source=OUT/'pilot_source/earth-representation'
    status=OUT/'pilot_status.json'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OBSERVATION_PROJECT_ROOT=str(ROOT),
        OBSERVATION_RELEASE=str(ROOT/'observation_model/outputs/phase0/release_local_v2'),
        OBSERVATION_CROSS_RELEASE=str(ROOT/'observation_model/outputs/phase0/cross_local_v2'))
    folder=OUT/'pilot1000'
    stages=[('training',['-m','observation_model.train_joint','--out',str(folder),'--steps','1000','--pilot']),
            ('validation_comparison',['-m','observation_model.evaluate_joint_pilot'])]
    for stage,args in stages:
        command=[sys.executable,'-u','-X','utf8',*args]
        if stage=='training' and (folder/'last.pt').exists():command.append('--resume')
        with (OUT/(stage+'.log')).open('a',encoding='utf-8') as stream:
            process=subprocess.Popen(command,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
            status.write_text(json.dumps(dict(status=stage,pid=process.pid,command=command),indent=2))
            code=process.wait()
        if code:
            status.write_text(json.dumps(dict(status='failed',stage=stage,returncode=code),indent=2));raise RuntimeError(stage)
    status.write_text(json.dumps(dict(status='PILOT_AND_VAL_COMPLETE',comparison=str(OUT/'pilot_comparison.json')),indent=2))


if __name__=='__main__':main()
