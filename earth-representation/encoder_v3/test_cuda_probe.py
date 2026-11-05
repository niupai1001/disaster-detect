import subprocess
import unittest
from unittest.mock import Mock,patch
from .cuda_probe import _probe_cuda


class CudaProbeChecks(unittest.TestCase):
    def test_native_failure_is_not_hidden_as_cuda_available(self):
        for code,output,expected in [(0,'CUDA_RUNTIME_OK 1.0',True),(3221225477,'native access violation',False),(0,'no success marker',False)]:
            child=Mock(returncode=code);child.communicate.return_value=(output,None)
            with patch('encoder_v3.cuda_probe.launch_python',return_value=child):result=_probe_cuda()
            self.assertEqual(result['available'],expected)
            self.assertEqual(result['exit_code'],code)

    def test_timeout_stops_only_the_owned_handle(self):
        child=Mock(returncode=1);child.communicate.side_effect=[subprocess.TimeoutExpired('owned',20),('interrupted',None)]
        with patch('encoder_v3.cuda_probe.launch_python',return_value=child):result=_probe_cuda()
        child.terminate.assert_called_once();child.kill.assert_not_called()
        self.assertTrue(result['timed_out']);self.assertFalse(result['available'])
