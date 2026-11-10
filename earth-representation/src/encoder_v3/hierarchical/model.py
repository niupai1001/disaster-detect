"""Parallel native summaries, sparse three-scale encoder, top-down MAE and EMA JEPA."""
from collections import OrderedDict
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from encoder_v3.shared_mae.model import SharedMemory, ConditionalConvolution
from encoder_v3.latent_response.geometry import position_features
from encoder_v3.tasks import task_rng, hidden_band_indices
from .geometry import partitions, area_map, neighborhood, sparse_edges

WIDTHS = (64, 96, 128)


class SparseAttention(nn.Module):
    def __init__(self, width, context_width=None):
        super().__init__()
        self.heads, self.width = 4, width
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(context_width or width, 2*width)
        self.out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))

    def attend(self, q, k, v, edges, valid, context_valid):
        qi, ki = edges
        allowed = valid[qi] & context_valid[ki]
        qi, ki = qi[allowed], ki[allowed]
        score = (q[qi]*k[ki]).sum(-1)/math.sqrt(self.width/4)
        index = qi[:,None].expand(-1,4)
        maximum = score.new_full((len(q),4),-torch.inf)
        maximum.scatter_reduce_(0,index,score.detach(),reduce='amax',include_self=True)
        weight = (score-maximum[qi]).exp()
        denominator = weight.new_zeros(len(q),4).index_add_(0,qi,weight)
        weight = weight/denominator[qi].clamp_min(1e-12)
        return v.new_zeros(len(q),4,self.width//4).index_add_(0,qi,weight[...,None]*v[ki]).flatten(1)

    def forward(self, x, context, edges, valid, context_valid):
        q = self.q(self.norm(x)).reshape(len(x),4,-1)
        k,v = self.kv(context).reshape(len(context),2,4,-1).unbind(1)
        outputs=[]
        for begin,end,index in edges:
            args=(q[begin:end],k,v,index,valid[begin:end],context_valid)
            outputs.append(checkpoint(self.attend,*args,use_reentrant=False) if torch.is_grad_enabled() else self.attend(*args))
        x = x+self.out(torch.cat(outputs))
        x = x+self.ff(x)
        return torch.where(valid[:,None],x,0.)


class GlobalBlock(nn.Module):
    """Exact global attention with bounded query chunks, including FP32 execution."""
    def __init__(self,width=128):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3)
        self.out=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))

    def forward(self,x,valid):
        ids=valid.nonzero().flatten()
        if not len(ids):return x*0
        selected=x[ids]
        q,k,v=self.qkv(self.norm(selected)).reshape(len(ids),3,4,-1).permute(1,2,0,3).unbind(0)
        outputs=[]
        for begin in range(0,len(ids),512):
            args=(q[:,begin:begin+512],k,v)
            outputs.append(checkpoint(F.scaled_dot_product_attention,*args,use_reentrant=False) if torch.is_grad_enabled() else F.scaled_dot_product_attention(*args))
        value=selected+self.out(torch.cat(outputs,1).transpose(0,1).flatten(1))
        value=value+self.ff(value)
        return torch.zeros_like(x).index_copy(0,ids,value)


class Encoder(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.memory, self.convolution = SharedMemory(), ConditionalConvolution()
        self.projection = nn.ModuleList(nn.Linear(33,w) for w in WIDTHS)
        self.fusion_key = nn.ModuleList(nn.Linear(w+32,1) for w in WIDTHS)
        self.fusion_value = nn.ModuleList(nn.Linear(w+32,w) for w in WIDTHS)
        self.position = nn.ModuleList(nn.Linear(50,w) for w in WIDTHS)
        self.branches = nn.ModuleList(nn.ModuleList(SparseAttention(w) for _ in range(depth)) for w in WIDTHS[:2])
        self.coarse = nn.ModuleList(GlobalBlock() for _ in range(depth))
        self.cross = nn.ModuleDict({name:SparseAttention(WIDTHS[q],WIDTHS[k]) for name,q,k in (
            ('f_to_m',1,0),('m_to_f',0,1),('m_to_c',2,1),('c_to_m',1,2))})
        self.norms = nn.ModuleList(nn.LayerNorm(w) for w in WIDTHS)

    def native(self, values, keep, kernel, bias, maps):
        clean = torch.where(keep,values,0.)
        # One convolution per channel/view, reused by all three independent aggregations.
        support = F.conv2d(keep[None,None].float(),values.new_ones(1,1,3,3),padding=1)
        feature = F.gelu(F.conv2d(clean[None,None],kernel,padding=1)*9/support.clamp_min(1)+bias[None,:,None,None])
        feature = feature[0].permute(1,2,0).reshape(-1,32)
        feature = torch.where(keep.flatten()[:,None],feature,0.)
        return tuple(value for mapping in maps for value in mapping.aggregate(feature,keep))

    def fuse(self,summary,support,condition,level):
        h=self.projection[level](torch.cat((summary,support[...,None]),-1))
        content=torch.cat((h,condition[:,None].expand(-1,h.shape[1],-1)),-1)
        present=support>0
        score=self.fusion_key[level](content).squeeze(-1).masked_fill(~present,-1e9)
        weights=score.softmax(0)*present
        weights=weights/weights.sum(0).clamp_min(1e-12)
        return (self.fusion_value[level](content)*weights[...,None]).sum(0)

    def forward(self, sample, layout, keeps):
        geo,parts,maps,edges,relations = layout
        conditions = self.memory([c.physical for c in sample.channels])
        kernels,biases = self.convolution(conditions['conv'])
        summaries = [[],[],[]]
        supports = [[],[],[]]
        for i,c in enumerate(sample.channels):
            def front(values,keep,kernel,bias,mapping=maps[i]):
                return self.native(values,keep,kernel,bias,mapping)
            args=(c.values,keeps[i],kernels[i],biases[i])
            values = checkpoint(front,*args,use_reentrant=False) if torch.is_grad_enabled() else front(*args)
            for level in range(3):
                summaries[level].append(values[level*2])
                supports[level].append(values[level*2+1])
        features,valids = [],[]
        for level,part in enumerate(parts):
            support = torch.stack(supports[level])
            chunks=[]
            for start in range(0,len(part),4096):
                summary=torch.stack([v[start:start+4096] for v in summaries[level]])
                def fuse(a,b,c,level=level):return self.fuse(a,b,c,level)
                args=(summary,support[:,start:start+4096],conditions['fusion'])
                chunks.append(checkpoint(fuse,*args,use_reentrant=False) if torch.is_grad_enabled() else fuse(*args))
            x=torch.cat(chunks)
            # Validity is support union (any), never a sum of overlapping channel areas.
            valid = (support>0).any(0)
            centers = torch.tensor(part.centers,device=x.device)+torch.tensor(geo.origin,device=x.device,dtype=torch.float64)
            x = x+self.position[level](position_features(centers,geo))
            features.append(torch.where(valid[:,None],x,0.));valids.append(valid)
        for level in range(2):
            for block in self.branches[level]:
                args=(features[level],features[level],edges[level],valids[level],valids[level])
                features[level] = checkpoint(block,*args,use_reentrant=False) if torch.is_grad_enabled() else block(*args)
        if not bool(valids[2].any()):
            raise ValueError('NO_VISIBLE_CONTEXT')
        for block in self.coarse:
            features[2] = checkpoint(block,features[2],valids[2],use_reentrant=False) if torch.is_grad_enabled() else block(features[2],valids[2])
        # One bottom-up and then top-down sweep after the independent branch stacks.
        for name,q,k,edge in (
            ('f_to_m',1,0,relations[0][1]),('m_to_c',2,1,relations[1][1]),
            ('c_to_m',1,2,relations[1][0]),('m_to_f',0,1,relations[0][0])):
            block=self.cross[name];args=(features[q],features[k],edge,valids[q],valids[k])
            features[q] = checkpoint(block,*args,use_reentrant=False) if torch.is_grad_enabled() else block(*args)
        return [norm(x) * valid[:,None] for norm,x,valid in zip(self.norms,features,valids)],valids,conditions


class HierarchicalMAE(nn.Module):
    def __init__(self, *, patch_metres, mask_ratio=.55, encoder_depth=2, lambda_jepa=0., ema_momentum=.996):
        super().__init__()
        if not 0<mask_ratio<1 or not 0<=ema_momentum<1 or lambda_jepa<0:
            raise ValueError('Invalid masking, EMA or JEPA weight')
        self.patch_metres,self.mask_ratio = patch_metres,mask_ratio
        self.lambda_jepa,self.ema_momentum = lambda_jepa,ema_momentum
        self.online = Encoder(encoder_depth)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.target.eval()
        self.decoder = nn.ModuleList([SparseAttention(w) for w in WIDTHS[:2]]+[GlobalBlock()])
        self.decoder_position = nn.ModuleList(nn.Linear(50,w) for w in WIDTHS)
        self.mask_tokens = nn.ParameterList(nn.Parameter(torch.randn(w)*.02) for w in WIDTHS)
        self.topdown = nn.ModuleList((nn.Linear(96,64),nn.Linear(128,96)))
        self.output = nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,1))
        self.predictor_context = nn.Linear(128,128)
        self.predictor_position = nn.Linear(50,128)
        self.predictor_scale = nn.Embedding(3,128)
        self.predictor = nn.Sequential(nn.LayerNorm(128),nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
        self.predictor_output = nn.ModuleList(nn.Linear(128,w) for w in WIDTHS)
        self.register_buffer('ema_updates',torch.zeros((),dtype=torch.long))
        self._layouts = OrderedDict()

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_ema(self):
        for target,online in zip(self.target.parameters(),self.online.parameters()):
            target.mul_(self.ema_momentum).add_(online,alpha=1-self.ema_momentum)
        for target,online in zip(self.target.buffers(),self.online.buffers()):
            target.copy_(online)
        self.ema_updates.add_(1)

    def layout(self,sample):
        grids=tuple(c.grid for c in sample.channels)
        device=sample.channels[0].values.device
        key=(grids,str(device))
        if key not in self._layouts:
            # Evict before building another full scene: two largest geometry maps must not coexist.
            self._layouts.clear()
            geo,parts,gsd=partitions(grids,self.patch_metres)
            unique={g:tuple(area_map(g,geo,p,device) for p in parts) for g in set(grids)}
            maps=[unique[g] for g in grids]
            edges=[neighborhood(parts[0],(gsd[0]*3,gsd[1]*3),device),
                   neighborhood(parts[1],(self.patch_metres,self.patch_metres),device)]
            import numpy as np
            relations=[]
            for level,p in enumerate(parts[:2]):
                index=np.stack((np.arange(len(p)),p.parent))
                relations.append((sparse_edges(index,len(p),device),sparse_edges(index[::-1],len(parts[level+1]),device),torch.tensor(p.parent,device=device)))
            self._layouts[key]=(geo,parts,maps,edges,relations)
            # Cache contains geometry only for the current observation, shared by all its tasks/views.
        self._layouts.move_to_end(key)
        return self._layouts[key]

    def available_tasks(self,sample):
        layout=self.layout(sample)
        occupied=torch.zeros(len(layout[1][2]),device=sample.channels[0].values.device,dtype=torch.bool)
        for c,maps in zip(sample.channels,layout[2]):
            support=maps[2].aggregate(c.values.new_zeros(c.values.numel(),1),c.valid)[1]
            occupied |= support>0
        kinds=(['A'] if int(occupied.sum())>=2 else [])+(['B1'] if sum(bool(c.valid.any()) for c in sample.channels)>=2 else [])
        if not kinds:
            raise ValueError('No valid reconstruction context')
        return kinds

    def masks(self,sample,layout,kind,seed,epoch):
        geo,parts,maps,_,_=layout
        device=sample.channels[0].values.device
        patch_mask=torch.zeros(len(parts[2]),device=device,dtype=torch.bool)
        eligible=[i for i,c in enumerate(sample.channels) if bool(c.valid.any())]
        hidden_bands=set()
        if kind=='A':
            occupied=torch.zeros_like(patch_mask)
            for c,mapping in zip(sample.channels,maps):
                occupied |= mapping[2].aggregate(c.values.new_zeros(c.values.numel(),1),c.valid)[1]>0
            ids=occupied.nonzero().flatten()
            if len(ids)<2:raise ValueError('A requires two occupied patches')
            count=min(len(ids)-1,max(1,round(len(ids)*self.mask_ratio)))
            rng=task_rng(seed,epoch,sample.identifier,'hierarchical_A')
            patch_mask[ids[rng.sample(range(len(ids)),count)]]=True
        elif kind=='B1':
            if len(eligible)<2:raise ValueError('B1 requires two valid channels')
            hidden_bands={eligible[i] for i in hidden_band_indices(len(eligible),seed=seed,update=epoch,observation_id=sample.identifier)}
        else:
            raise ValueError('A or B1 required')
        keeps,targets=[],[]
        for i,(c,mapping) in enumerate(zip(sample.channels,maps)):
            hidden=mapping[2].hidden(patch_mask).reshape(c.valid.shape) if kind=='A' else torch.full_like(c.valid,i in hidden_bands)
            keeps.append(c.valid & ~hidden);targets.append(c.valid & hidden)
        return keeps,targets,patch_mask

    def decode(self,encoded,valids,layout,query_valid):
        geo,parts,_,edges,relations=layout
        decoded=[None,None,None]
        for level in (2,1,0):
            x=encoded[level]
            centers=torch.tensor(parts[level].centers,device=x.device)+torch.tensor(geo.origin,device=x.device,dtype=torch.float64)
            x=torch.where(valids[level][:,None],x,self.mask_tokens[level])+self.decoder_position[level](position_features(centers,geo))
            if level<2:
                parent=relations[level][2]
                x=x+self.topdown[level](decoded[level+1][parent])
            valid=query_valid[level]
            if level==2:
                block=self.decoder[level]
                decoded[level]=checkpoint(block,x,valid,use_reentrant=False) if torch.is_grad_enabled() else block(x,valid)
            else:
                block=self.decoder[level];args=(x,x,edges[level],valid,valid)
                decoded[level]=checkpoint(block,*args,use_reentrant=False) if torch.is_grad_enabled() else block(*args)
        return decoded

    def forward(self,samples,*,kind='A',seed=17,epoch=0,return_predictions=False):
        losses,details=[],[]
        for sample in samples:
            layout=self.layout(sample)
            keeps,targets,patch_mask=self.masks(sample,layout,kind,seed,epoch)
            encoded,valids,conditions=self.online(sample,layout,keeps)
            query_valid=[]
            for level,part in enumerate(layout[1]):
                supported=torch.zeros(len(part),device=encoded[0].device,dtype=torch.bool)
                for c,mapping in zip(sample.channels,layout[2]):
                    supported |= mapping[level].aggregate(c.values.new_zeros(c.values.numel(),1),c.valid)[1]>0
                query_valid.append(supported)
            decoded=self.decode(encoded,valids,layout,query_valid)
            channel_losses,predictions=[],[]
            for i,c in enumerate(sample.channels):
                context=conditions['output'][i].expand(len(decoded[0]),-1)
                field=self.output(torch.cat((decoded[0],context),-1)).squeeze(-1)
                predicted=layout[2][i][0].native(field).reshape(c.values.shape)
                mask=targets[i]
                if bool(mask.any()):
                    error=torch.where(mask,predicted,0.)-torch.where(mask,c.values,0.)
                    channel_losses.append(error.square().sum()/mask.sum())
                predictions.append(predicted if return_predictions else None)
            if not channel_losses:raise ValueError('NO_VALID_TARGETS')
            mae=torch.stack(channel_losses).mean()
            jepa=mae.new_zeros(())
            scale_losses,scale_counts=[],[]
            target_variance=[]
            if self.lambda_jepa>0 and kind=='A':
                with torch.no_grad():
                    target,tvalid,_=self.target(sample,layout,[c.valid for c in sample.channels])
                context=self.predictor_context(encoded[2][valids[2]].mean(0))
                for level,part in enumerate(layout[1]):
                    mask=patch_mask[torch.tensor(part.patch,device=mae.device)] & tvalid[level]
                    centers=torch.tensor(part.centers,device=mae.device)+torch.tensor(layout[0].origin,device=mae.device,dtype=torch.float64)
                    query=self.predictor_position(position_features(centers,layout[0]))+context+self.predictor_scale.weight[level]
                    predicted=self.predictor_output[level](self.predictor(query))
                    if bool(mask.any()):
                        truth=F.layer_norm(target[level][mask],(WIDTHS[level],))
                        value=F.smooth_l1_loss(predicted[mask],truth)
                        scale_losses.append(value);scale_counts.append(int(mask.sum()))
                        target_variance.append(float(truth.std(0,unbiased=False).mean()))
                if not scale_losses:raise ValueError('NO_JEPA_TARGETS')
                jepa=torch.stack(scale_losses).mean()
            loss=mae+self.lambda_jepa*jepa
            losses.append(loss)
            details.append(dict(id=sample.identifier,task=kind,mae=float(mae.detach()),jepa=float(jepa.detach()),
                jepa_scales=[float(v.detach()) for v in scale_losses],jepa_counts=scale_counts,target_std=target_variance,
                tokens=[len(p) for p in layout[1]],visible_tokens=[int(v.sum()) for v in valids],
                attention_edges=[sum(chunk[2].shape[1] for chunk in e) for e in layout[3]],targets=sum(int(t.sum()) for t in targets),
                predictions=predictions if return_predictions else None,target_masks=targets if return_predictions else None,
                online_features=encoded if return_predictions else None))
        return torch.stack(losses),details
