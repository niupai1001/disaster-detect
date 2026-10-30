"""Same consumer tensors and forward; verify the full frozen encoder every update."""
from landslide_seg.models.use_next_p2.readout import ScaleReadout
from landslide_seg.models.use_next_v2.readout import state_digest


class GuardedReadout(ScaleReadout):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.frozen_encoder_sha256=state_digest(self.use)

    def after_optimizer(self,step,total):
        if state_digest(self.use)!=self.frozen_encoder_sha256:
            raise ValueError('Frozen full USE changed after optimizer update; do not save or continue')
