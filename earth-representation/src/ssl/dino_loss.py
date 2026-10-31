"""DINO loss and the teacher centring buffer."""

import torch
import torch.nn.functional as F


class DINOLoss(torch.nn.Module):
    def __init__(self, out_dim=4096, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum

    def forward(self, student_out, teacher_out, n_global=2):
        """student_out: [n_views, B, D]; teacher_out: [n_global, B, D]."""
        student = student_out.chunk(student_out.shape[0])
        teacher = teacher_out.chunk(teacher_out.shape[0])
        total, terms = 0.0, 0
        n_samples = teacher_out.shape[1]
        for ti, t in enumerate(teacher):
            t = F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for si, s in enumerate(student):
                if si == ti:
                    continue
                total = total + torch.sum(-t * F.log_softmax(s / self.student_temp, dim=-1))
                terms += 1
        # normalise by pairs * batch, otherwise the value scales with batch size
        loss = total / max(terms * n_samples, 1)
        self.update_center(torch.cat(teacher, dim=0))
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        """teacher_out: [n_global, B, D]; one centre vector per prototype.

        Averaging only over the view axis leaves B in the buffer, which then
        breaks broadcasting on the next batch of a different size.
        """
        batch_center = teacher_out.mean(dim=(0, 1)).unsqueeze(0)
        self.center = self.center * self.center_momentum + \
            batch_center * (1 - self.center_momentum)
