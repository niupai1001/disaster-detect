"""Bounded, reusable exact-area maps between native pixels and metric patch slots."""
from collections import OrderedDict
from dataclasses import dataclass
import math
import torch
from encoder_v3.latent_response.geometry import FullGeometry, native_cells


@dataclass
class AreaMap:
    source: torch.Tensor
    destination: torch.Tensor
    area: torch.Tensor
    pixels: int
    slots: int
    pixel_area: float
    slot_area: float

    def aggregate(self, features, keep):
        # features [C,N,D], keep [C,N]; native values never get resampled in storage.
        weight = self.area.to(features.dtype)[None]*keep[:, self.source]
        support = features.new_zeros(features.shape[0], self.slots).index_add(1, self.destination, weight)
        total = features.new_zeros(features.shape[0], self.slots, features.shape[-1]).index_add(
            1, self.destination, features[:, self.source]*weight[..., None])
        return total/support.clamp_min(torch.finfo(features.dtype).tiny)[..., None], support/self.slot_area

    def native_prediction(self, field):
        # [P,R*R] predictions are area-averaged back onto the ORIGINAL pixel footprint.
        return self.native_predictions(field[None])[0]

    def native_predictions(self, fields):
        # Same geometry, independent channels: [C,P,R*R] -> [C,N_native].
        return fields.new_zeros(fields.shape[0],self.pixels).index_add(1,self.source,
            fields.flatten(1)[:,self.destination]*self.area.to(fields.dtype)[None]/self.pixel_area)

    def hidden_pixels(self, patch_mask, resolution):
        hidden = patch_mask[self.destination//(resolution*resolution)]
        hits = torch.zeros(self.pixels, device=self.source.device, dtype=torch.int32)
        hits.index_add_(0, self.source, hidden.to(torch.int32))
        return hits > 0


def metric_geometry(channels, patch_metres):
    if not math.isfinite(patch_metres) or patch_metres <= 0:
        raise ValueError('Positive metric patch size required')
    original = FullGeometry.from_channels(channels)
    w, h = original.extent
    return FullGeometry(original.grids, original.origin, original.extent, patch_metres,
                        (math.ceil(h/patch_metres), math.ceil(w/patch_metres)))


class AreaCache:
    def __init__(self, limit_bytes=1024*1024**2):
        self.limit = limit_bytes
        self.items = OrderedDict()
        self.bytes = 0

    def get(self, grid, geometry, resolution, device):
        # Area weights depend on relative offsets, not a scene's absolute map coordinates.
        # Reuse equal layouts across scenes while retaining genuine fractional alignment.
        offset = (grid.affine[2]-geometry.origin[0], grid.affine[5]-geometry.origin[1])
        key = (grid.shape, grid.gsd_xy, offset, geometry.shape, geometry.patch_size, resolution, str(device))
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key][0]
        # CPU geometry construction avoids repeated device synchronization for variable lengths.
        boxes = native_cells(grid, 'cpu')
        origin = boxes.new_tensor(geometry.origin)
        size = geometry.patch_size/resolution
        lo = (boxes[:, :2]-origin)/size
        hi = (boxes[:, 2:]-origin)/size
        # Affine arithmetic at large map coordinates can put an aligned edge ~1e-10 off-grid.
        lo = torch.where((lo-lo.round()).abs()<1e-8, lo.round(), lo)
        hi = torch.where((hi-hi.round()).abs()<1e-8, hi.round(), hi)
        low, high = lo.floor().long(), hi.ceil().long()-1
        spans = high-low+1
        lengths = spans.prod(-1)
        if bool((lengths < 1).any()) or int(lengths.max()) > 4096:
            raise ValueError('Unsupported native-to-patch footprint expansion')
        source = torch.repeat_interleave(torch.arange(len(boxes)), lengths)
        offset = torch.arange(int(lengths.sum()))-torch.repeat_interleave(lengths.cumsum(0)-lengths, lengths)
        nx = spans[source, 0]
        ix, iy = low[source, 0]+offset%nx, low[source, 1]+offset//nx
        lower = origin+torch.stack((ix, iy), -1)*size
        area = (torch.minimum(boxes[source, 2:], lower+size)-torch.maximum(boxes[source, :2], lower)).clamp_min(0).prod(-1)
        destination = ((iy//resolution)*geometry.shape[1]+ix//resolution)*resolution**2+(iy%resolution)*resolution+ix%resolution
        slots = geometry.shape[0]*geometry.shape[1]*resolution**2
        if bool((destination < 0).any()) or bool((destination >= slots).any()):
            raise ValueError('Native footprint extends beyond immutable geometry')
        result = AreaMap(source.to(device), destination.to(device), area.to(device), len(boxes), slots,
                         grid.gsd_xy[0]*grid.gsd_xy[1], size*size)
        cost = sum(t.numel()*t.element_size() for t in (result.source, result.destination, result.area))
        if cost <= self.limit:
            while self.items and self.bytes+cost > self.limit:
                _, (_, n) = self.items.popitem(last=False)
                self.bytes -= n
            self.items[key] = (result, cost)
            self.bytes += cost
        return result
