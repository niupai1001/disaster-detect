"""DINO-style EMA teacher."""

import copy
import math

import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    """Holds the teacher weights and the momentum schedule."""

    def __init__(self, student, base_momentum=0.99, total_steps=1000, cap=0.999):
        super().__init__()
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.base = base_momentum
        self.cap = cap
        self.total = max(total_steps, 1)
        self.step = 0

    def momentum(self):
        """Ramp base -> cap; never 1.0, or the teacher stops tracking the student."""
        m = self.base + (self.cap - self.base) * (
            1.0 - math.cos(math.pi * min(self.step, self.total) / self.total)) / 2.0
        return min(m, self.cap)

    @torch.no_grad()
    def update_from(self, student):
        """teacher = m * teacher + (1 - m) * student, with a cosine warmup.

        Buffers (BatchNorm running statistics) are copied as well: the teacher
        runs in eval mode, so without this it would keep the statistics of the
        randomly initialised student forever.
        """
        m = self.momentum()
        for ps, pt in zip(student.parameters(), self.teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bs, bt in zip(student.buffers(), self.teacher.buffers()):
            if bt.dtype.is_floating_point:
                bt.mul_(m).add_(bs.detach(), alpha=1.0 - m)
            else:
                bt.copy_(bs)
        self.step += 1
        return m


