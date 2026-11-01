"""DINO over two observation groups, with optional cross-sensor targets."""

import torch
import torch.nn.functional as F


class ViewMetadataMixin:
    def set_view_meta(self, centres, gsds, masks, n_base):
        self._view_meta = (centres, gsds, masks)
        self._view_cursor = 0
        self._n_base = n_base

    def set_meta(self, *args, **kwargs):
        self._view_meta = None
        return super().set_meta(*args, **kwargs)

    def _meta(self, x):
        if getattr(self, "_view_meta", None) is None:
            return super()._meta(x)
        assert x.shape[0] % self._n_base == 0
        count = x.shape[0] // self._n_base
        start = self._view_cursor
        self._view_cursor += count
        assert self._view_cursor <= len(self._view_meta[0])
        return tuple(torch.cat(values[start:start + count]) for values in self._view_meta)


def observation_dino_loss(student_output, teacher_output, center, teacher_temp,
                          student_temp=.1, cross_sensor=False):
    """Four global + four local student views; four global teacher views.

    Order: A-global0/1, B-global0/1, A-local0/1, B-local0/1.
    Within-observation DINO is exactly the mean of the two ordinary DINO losses.
    """
    students = (student_output / student_temp).chunk(8)
    teachers = F.softmax((teacher_output - center) / teacher_temp, dim=-1).detach().chunk(4)
    groups = (0, 0, 1, 1, 0, 0, 1, 1)
    terms = []
    for i, target in enumerate(teachers):
        for j, output in enumerate(students):
            if i == j or (not cross_sensor and groups[i] != groups[j]):
                continue
            terms.append(-(target * F.log_softmax(output, dim=-1)).sum(-1).mean())
    return torch.stack(terms).mean()
