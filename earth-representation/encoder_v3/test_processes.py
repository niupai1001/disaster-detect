import json
import subprocess
import sys
import unittest
from .processes import launch_python


class ProcessChecks(unittest.TestCase):
    def test_handle_is_actual_interpreter_and_prefix_is_preserved(self):
        child=launch_python(['-c','import json,os,sys;print(json.dumps(dict(pid=os.getpid(),prefix=sys.prefix)))'],stdout=subprocess.PIPE)
        try:output,_=child.communicate(timeout=20)
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=5)
        self.assertEqual(child.returncode,0)
        value=json.loads(output);self.assertEqual(value['pid'],child.pid);self.assertEqual(value['prefix'],sys.prefix)
