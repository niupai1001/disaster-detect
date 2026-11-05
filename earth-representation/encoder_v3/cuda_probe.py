"""Probe native GPU initialization in an isolated, bounded interpreter."""
from functools import lru_cache
import subprocess
from .processes import launch_python


def _probe_cuda():
    child=launch_python(['-c',"import torch; x=torch.ones(1,device='cuda:0'); print('CUDA_RUNTIME_OK',float(x.sum()),flush=True)"],stdout=subprocess.PIPE)
    timed_out=False
    try:output,_=child.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        timed_out=True;child.terminate()
        try:output,_=child.communicate(timeout=5)
        except subprocess.TimeoutExpired:child.kill();output,_=child.communicate(timeout=5)
    return dict(available=not timed_out and child.returncode==0 and 'CUDA_RUNTIME_OK' in output,exit_code=child.returncode,timed_out=timed_out,diagnostic=output[-5000:])


@lru_cache(maxsize=1)
def probe_cuda():return _probe_cuda()
