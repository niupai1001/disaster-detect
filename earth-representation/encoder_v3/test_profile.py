import unittest
from .profile_task import scoped


class ProfileChecks(unittest.TestCase):
    def test_scoped_call_preserves_result_and_exception(self):
        timings={}
        self.assertEqual(scoped(lambda value:value+1,'example',timings)(2),3)
        def fail():raise ValueError('original error')
        with self.assertRaisesRegex(ValueError,'original error'):scoped(fail,'example',timings)()
        self.assertEqual(timings['example']['calls'],2)
        self.assertGreaterEqual(timings['example']['wall_seconds'],0)
