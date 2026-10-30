"""UUID-based exclusive lock shared by engineering and independent audit."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


@contextmanager
def claim(device):
    index=int(device.split(':')[-1])
    if os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Use explicit physical CUDA indices without remapping')
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
    mapping={int(line.split(',')[0]):line.split(',')[1].strip() for line in lines}
    uuid=mapping[index];path=ROOT/'outputs/use_next_v2/engineering/device_locks'/f'{uuid}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(dict(pid=os.getpid(),device=device,uuid=uuid),f)
    try:yield uuid
    finally:path.unlink()
