"""Launch the actual Python worker, preserving the existing virtual environment."""
import os
import subprocess
import sys


def launch_python(arguments,*,stdout=None):
    environment=os.environ.copy();executable=sys.executable
    if sys.platform=='win32':
        # The venv redirector has a different PID from Python and cannot bound it.
        executable=sys._base_executable
        environment['__PYVENV_LAUNCHER__']=sys.executable
    environment['PYTHONIOENCODING']='utf-8'
    return subprocess.Popen([executable,'-X','faulthandler',*arguments],env=environment,stdout=stdout,stderr=subprocess.STDOUT if stdout is not None else None,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
