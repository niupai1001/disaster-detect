import unittest
from encoder_v3.communication_probe import join_bounded


class ProcessFixture:
    def __init__(self):self.alive=True;self.terminated=False
    def is_alive(self):return self.alive
    def terminate(self):self.terminated=True;self.alive=False
    def join(self,timeout):pass
    def kill(self):self.alive=False


class CommunicationChecks(unittest.TestCase):
    def test_controller_deadline_reaps_only_owned_handles(self):
        class Context:
            processes=[ProcessFixture(),ProcessFixture()]
            def join(self,timeout):return False
        context=Context()
        self.assertFalse(join_bounded(context,0))
        self.assertTrue(all(p.terminated for p in context.processes))
