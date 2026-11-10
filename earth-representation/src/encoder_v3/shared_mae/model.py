"""Single-token metric-patch Encoder and global MAE Decoder; no pixel retrieval."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from encoder_v3.tasks import task_rng, hidden_band_indices
from encoder_v3.latent_response.geometry import position_features
from .geometry import AreaCache, metric_geometry
from .metadata import QUANTITIES, POLARIZATIONS


@dataclass
class NativeChannel:
    values: torch.Tensor
    valid: torch.Tensor
    grid: object
    physical: object
    name: str


@dataclass
class Sample:
    identifier: str
    role: str
    channels: list


class SharedMemory(nn.Module):
    """Memory slots are shared across sensors, including unseen sensor names."""
    def __init__(self):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(8, 32)*.02)
        self.key = nn.Linear(32, 32, bias=False)
        self.value = nn.Linear(32, 32, bias=False)
        self.modality = nn.Embedding(2, 4)
        self.quantity = nn.Embedding(len(QUANTITIES), 4)
        self.polarization = nn.Embedding(len(POLARIZATIONS), 4)
        self.query = nn.Sequential(nn.Linear(24, 64), nn.GELU(), nn.Linear(64, 32))
        self.module_seed = nn.ParameterDict({name: nn.Parameter(torch.randn(32)*.02) for name in ('conv', 'fusion', 'output')})

    def forward(self, specs):
        encoded = [s.encoded() for s in specs]
        numeric = self.slots.new_tensor([x[0] for x in encoded])
        categories = torch.tensor([x[1] for x in encoded], device=numeric.device, dtype=torch.long)
        base = self.query(torch.cat((numeric, self.modality(categories[:, 0]), self.quantity(categories[:, 1]),
                                     self.polarization(categories[:, 2])), -1))
        keys, values = self.key(self.slots), self.value(self.slots)
        return {name: torch.softmax((base+seed)@keys.T/math.sqrt(32), -1)@values for name, seed in self.module_seed.items()}


class ConditionalConvolution(nn.Module):
    def __init__(self, width=32, bases=8):
        super().__init__()
        self.base = nn.Parameter(torch.empty(width, 1, 3, 3))
        self.bases = nn.Parameter(torch.randn(bases, width, 1, 3, 3)*.02)
        self.bias = nn.Parameter(torch.zeros(width))
        self.bias_bases = nn.Parameter(torch.randn(bases, width)*.01)
        self.coefficients = nn.Linear(32, bases)
        nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))

    def forward(self, condition):
        alpha = self.coefficients(condition).softmax(-1)
        return self.base[None]+torch.einsum('cb,bdkhw->cdkhw', alpha, self.bases), self.bias[None]+alpha@self.bias_bases


class GlobalTransformer(nn.Module):
    """Global attention per scene; padding never allows scenes to communicate."""
    def __init__(self, width, depth):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(width, 4, width*4, dropout=0.,
                                                                batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, features, indices):
        lengths = [len(x) for x in features]
        if not lengths or min(lengths) < 1:
            raise ValueError('Global attention requires nonempty scenes')
        values = nn.utils.rnn.pad_sequence(features, batch_first=True)
        padding = torch.arange(values.shape[1], device=values.device)[None] >= torch.tensor(lengths, device=values.device)[:, None]
        for layer in self.layers:
            values = layer(values, src_key_padding_mask=padding)
        values = self.norm(values)
        return [values[i, :length] for i, length in enumerate(lengths)]


class SharedMAE(nn.Module):
    def __init__(self, *, patch_metres, width=128, layout=8, encoder_depth=4, decoder_depth=2,
                 mask_ratio=.75, decoder_width=64, frontend_tile_rows=128, frontend_group_pixels=262144):
        super().__init__()
        if not math.isfinite(patch_metres) or patch_metres <= 0:
            raise ValueError('Explicit positive patch_metres L required; no implicit 80m default')
        if min(width, decoder_width) < 4 or width % 4 or decoder_width % 4 or layout < 1 or not 0 < mask_ratio < 1 or min(encoder_depth, decoder_depth, frontend_tile_rows, frontend_group_pixels) < 1:
            raise ValueError('Invalid model dimensions or masking ratio')
        self.patch_metres, self.layout, self.mask_ratio = patch_metres, layout, mask_ratio
        self.frontend_tile_rows, self.frontend_group_pixels = frontend_tile_rows, frontend_group_pixels
        self.memory = SharedMemory()
        self.convolution = ConditionalConvolution()
        self.patch_embedding = nn.Linear(layout*layout*33, width)
        self.fusion_key = nn.Linear(width+32, width)
        self.fusion_value = nn.Linear(width+32, width)
        self.fusion_query = nn.Parameter(torch.randn(width)*.02)
        self.fusion = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.position = nn.Linear(50, width)
        self.encoder = GlobalTransformer(width, encoder_depth)
        self.decoder_embedding = nn.Linear(width, decoder_width)
        self.decoder_position = nn.Linear(50, decoder_width)
        self.mask_token = nn.Parameter(torch.randn(decoder_width)*.02)
        self.decoder = GlobalTransformer(decoder_width, decoder_depth)
        self.output = nn.Sequential(nn.Linear(decoder_width+32, decoder_width), nn.GELU(), nn.Linear(decoder_width, layout*layout))
        self.area_cache = AreaCache()

    def _patch_features(self, values, keep, kernels, biases, mapping, geometry):
        count, height, width = values.shape
        clean = torch.where(keep, values, 0.)
        # Tiles preserve the 1-pixel halo; hidden values have already been removed.
        features = []
        for start in range(0, height, self.frontend_tile_rows):
            end = min(height, start+self.frontend_tile_rows)
            low, high = max(0, start-1), min(height, end+1)
            support = F.conv2d(keep[:, None, low:high].to(values.dtype), values.new_ones(1, 1, 3, 3), padding=1)
            result = F.conv2d(clean[:, low:high].reshape(1, count, high-low, width),
                              kernels.reshape(count*32, 1, 3, 3), padding=1, groups=count)
            result = result.reshape(count, 32, high-low, width)*(9./support.clamp_min(1))+biases[:, :, None, None]
            features.append(F.gelu(result[:, :, start-low:end-low]))
        features = torch.cat(features, 2).permute(0, 2, 3, 1).reshape(count, -1, 32)
        features = torch.where(keep.flatten(1)[..., None], features, 0.)
        cells, support = mapping.aggregate(features, keep.flatten(1))
        patches = geometry.shape[0]*geometry.shape[1]
        cells = cells.reshape(count, patches, self.layout**2*32)
        support = support.reshape(count, patches, self.layout**2)
        return self.patch_embedding(torch.cat((cells, support), -1)), support.any(-1)

    def _layout(self, sample):
        channels = sample.channels
        if not channels:
            raise ValueError('Nonempty native observation required')
        geometry = metric_geometry(channels, self.patch_metres)
        device = channels[0].values.device
        # Large maps may exceed the persistent cache; still share one map per grid in this view.
        unique_maps = {}
        maps = []
        valid_by_grid = {}
        for c in channels:
            if c.grid not in unique_maps:
                unique_maps[c.grid] = self.area_cache.get(c.grid, geometry, self.layout, device)
                valid_by_grid[c.grid] = c.valid.clone()
            else:
                valid_by_grid[c.grid] |= c.valid
            maps.append(unique_maps[c.grid])
        total = geometry.shape[0]*geometry.shape[1]
        occupied = torch.zeros(total, device=device, dtype=torch.int32)
        for grid, valid in valid_by_grid.items():
            mapping = unique_maps[grid]
            occupied.index_add_(0, mapping.destination//self.layout**2, valid.flatten()[mapping.source].int())
        valid_ids = (occupied > 0).nonzero().flatten()
        valid_flags = torch.stack([c.valid.any() for c in channels]).tolist()
        eligible = [i for i, flag in enumerate(valid_flags) if flag]
        if not len(valid_ids):
            raise ValueError('NO_VALID_PIXELS')
        return geometry, maps, valid_ids, eligible

    def available_tasks(self, sample):
        _, _, patches, eligible = self._layout(sample)
        kinds = (['A'] if len(patches) >= 2 else [])+(['B1'] if len(eligible) >= 2 else [])
        if not kinds:raise ValueError('Observation has neither spatial nor channel reconstruction context')
        return kinds

    def _prepare(self, sample, kind, seed, epoch):
        channels = sample.channels
        if kind not in ('A', 'B1'):raise ValueError('A/B1 task required')
        geometry, maps, valid_ids, eligible = self._layout(sample)
        device = channels[0].values.device
        total = geometry.shape[0]*geometry.shape[1]
        patch_mask = torch.zeros(total, device=device, dtype=torch.bool)
        band_hidden = set()
        if kind == 'A':
            if len(valid_ids) < 2:
                raise ValueError('A requires at least two occupied patches')
            rng = task_rng(seed, epoch, sample.identifier, 'shared_mae_A')
            hide_count = min(len(valid_ids)-1, max(1, round(len(valid_ids)*self.mask_ratio)))
            choices = rng.sample(range(len(valid_ids)), hide_count)
            patch_mask[valid_ids[torch.tensor(choices, device=device)]] = True
        else:
            if len(eligible) < 2:
                raise ValueError('B1 requires two valid channels; single-channel observations support A only')
            band_hidden = {eligible[i] for i in hidden_band_indices(len(eligible), seed=seed, update=epoch, observation_id=sample.identifier)}
        keeps, targets = [], []
        hidden_by_grid = {}
        for i, (c, mapping) in enumerate(zip(channels, maps)):
            if kind == 'A':
                if c.grid not in hidden_by_grid:
                    hidden_by_grid[c.grid] = mapping.hidden_pixels(patch_mask, self.layout).reshape(c.values.shape)
                hidden = hidden_by_grid[c.grid]
            else:
                hidden = torch.full_like(c.valid, i in band_hidden)
            keeps.append(c.valid & ~hidden)
            targets.append(c.valid & hidden)
        conditions = self.memory([c.physical for c in channels])
        kernels, biases = self.convolution(conditions['conv'])
        groups = {}
        keep_flags = torch.stack([keep.any() for keep in keeps]).tolist()
        for i, c in enumerate(channels):
            if keep_flags[i]:groups.setdefault(c.grid, []).append(i)
        embedded, present, order = [], [], []
        for grid, indices in groups.items():
            capacity = max(1, self.frontend_group_pixels//(grid.shape[0]*grid.shape[1]))
            for start in range(0, len(indices), capacity):
                selected = indices[start:start+capacity]
                values = torch.stack([channels[i].values for i in selected])
                keep = torch.stack([keeps[i] for i in selected])
                mapping = maps[selected[0]]
                def front(x, m, w, b, mapping=mapping):
                    return self._patch_features(x, m, w, b, mapping, geometry)
                arguments = (values, keep, kernels[selected], biases[selected])
                estimated_bytes = values.numel()*32*values.element_size()
                h, active = checkpoint(front, *arguments, use_reentrant=False) if torch.is_grad_enabled() and estimated_bytes > 128*1024**2 else front(*arguments)
                embedded.append(h);present.append(active);order += selected
        if not embedded:
            raise ValueError('NO_VISIBLE_CONTEXT after native-footprint masking')
        h, available = torch.cat(embedded).transpose(0, 1), torch.cat(present).T
        visible_ids = available.any(-1).nonzero().flatten()
        content = torch.cat((h[visible_ids], conditions['fusion'][order][None].expand(len(visible_ids), -1, -1)), -1)
        q = self.fusion_query[None, None].expand(len(visible_ids), 1, -1)
        token = self.fusion(q, self.fusion_key(content), self.fusion_value(content),
                            key_padding_mask=~available[visible_ids], need_weights=False)[0][:, 0]
        index, centers = geometry.patch_centers(device)
        location = position_features(centers, geometry).to(token.dtype)
        pos = self.position(location)
        return dict(sample=sample, geometry=geometry, maps=maps, conditions=conditions, valid_ids=valid_ids,
                    visible_ids=visible_ids, token=token+pos[visible_ids], position=self.decoder_position(location), index=index, targets=targets,
                    patch_mask=patch_mask, kind=kind)

    def forward(self, samples, *, kind='A', seed=17, epoch=0, return_predictions=False):
        prepared = [self._prepare(s, kind, seed, epoch) for s in samples]
        encoded = self.encoder([p['token'] for p in prepared], [p['index'][p['visible_ids']] for p in prepared])
        decoder_input = []
        for p, features in zip(prepared, encoded):
            projected = self.decoder_embedding(features)
            x = self.mask_token.to(projected.dtype)[None].expand(len(p['index']), -1).clone()
            x[p['visible_ids']] = projected
            decoder_input.append((x+p['position'])[p['valid_ids']])
        decoded = self.decoder(decoder_input, [p['index'][p['valid_ids']] for p in prepared])
        losses, details = [], []
        for p, features in zip(prepared, decoded):
            channel_losses, zero_losses = [], []
            predictions = [None]*len(p['sample'].channels)
            target_flags = torch.stack([mask.any() for mask in p['targets']]).tolist()
            groups = {}
            for i, channel in enumerate(p['sample'].channels):
                if target_flags[i]:groups.setdefault(channel.grid,[]).append(i)
            # Batch shared-grid channel heads; bound temporary C*P independently of scene size.
            capacity=max(1,131072//len(features))
            for indices in groups.values():
                for start in range(0,len(indices),capacity):
                    chosen=indices[start:start+capacity]
                    condition=p['conditions']['output'][chosen][:,None].expand(-1,len(features),-1)
                    spatial=features[None].expand(len(chosen),-1,-1)
                    field=self.output(torch.cat((spatial,condition),-1)).float()
                    full=field.new_zeros(len(chosen),len(p['index']),self.layout**2).index_copy(1,p['valid_ids'],field)
                    predicted=p['maps'][chosen[0]].native_predictions(full)
                    mask=torch.stack([p['targets'][i].flatten() for i in chosen])
                    values=torch.stack([p['sample'].channels[i].values.flatten() for i in chosen]).float()
                    target=torch.where(mask,values,0.)
                    error=torch.where(mask,predicted,0.)-target
                    counts=mask.sum(-1)
                    channel_losses.append(error.square().sum(-1)/counts)
                    zero_losses.append((target.square().sum(-1)/counts).detach())
                    if return_predictions:
                        for j,i in enumerate(chosen):predictions[i]=predicted[j].reshape(p['sample'].channels[i].values.shape)
            if not channel_losses:
                raise ValueError('NO_VALID_TARGETS')
            losses.append(torch.cat(channel_losses).mean())
            details.append(dict(id=p['sample'].identifier, task=kind, patches=len(p['valid_ids']),
                encoder_tokens=len(p['visible_ids']), targets=int(torch.stack([m.sum() for m in p['targets']]).sum()),
                zero_loss=float(torch.cat(zero_losses).mean()),
                predictions=predictions if return_predictions else None, target_masks=p['targets'] if return_predictions else None))
        return torch.stack(losses), details
