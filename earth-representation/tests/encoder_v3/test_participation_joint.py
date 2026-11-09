from tests.encoder_v3 import test_adaptive_joint as base
from encoder_v3.participation_encoder import ParticipationEncoder,prepare_participation_task
from encoder_v3.participation_layout import AreaEquivalentHillRate


class ParticipationJointChecks(base.AdaptiveJointChecks):
    # Explicit candidate curve fixture; no frozen scientific coefficients.
    def curve(self):return AreaEquivalentHillRate(.8092248954506467,4.010953026221307,1.219268281571556)
    def encoder(self):return ParticipationEncoder(rate_curve=self.curve(),selection_seed=0,hyper_hidden=64)
    def plan(self,task):return prepare_participation_task(task,rate_curve=self.curve(),selection_seed=0,origin_xy=(0,0),chunk_pixels=7,cell_indexed=True)
