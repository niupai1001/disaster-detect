"""Parallel native summaries, sparse three-scale encoder, top-down MAE and EMA JEPA."""
from collections import OrderedDict
import copy
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from encoder_v3.shared_mae.model import SharedMemory, ConditionalConvolution
from encoder_v3.latent_response.geometry import position_features
from encoder_v3.tasks import task_rng, hidden_band_indices
from .geometry import partitions, area_map, neighborhood, sparse_edges, metric_geometry

WIDTHS = (64, 96, 128)


@dataclass
class ScaleTokens:
    features: torch.Tensor
    coordinates: torch.Tensor
    support_ratio: torch.Tensor
    valid: torch.Tensor
    parents: torch.Tensor | None
    children: object
    geometry: object
    channel_indices: tuple


@dataclass
class EncoderOutput:
    features: list
    valid: list
    conditions: dict
    tokens: list

    def __iter__(self):return iter((self.features,self.valid,self.conditions))
    def __getitem__(self,index):return (self.features,self.valid,self.conditions)[index]


def execute(function,*args,large=False):
    return checkpoint(function,*args,use_reentrant=False) if torch.is_grad_enabled() and large else function(*args)


def attention(query,key,value,attn_mask=None):
    if query.device.type=='cpu':
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(query,key,value,attn_mask=attn_mask)
    return F.scaled_dot_product_attention(query,key,value,attn_mask=attn_mask)


def masked_softmax(scores,mask,dimension):
    masked=scores.masked_fill(~mask,-torch.inf)
    safe=torch.where(mask.any(dimension,keepdim=True),masked,0.)
    return safe.softmax(dimension)*mask


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
        same=x is context and valid is context_valid
        x=torch.where(valid[:,None],x,0.)
        context=x if same else torch.where(context_valid[:,None],context,0.)
        q = self.q(self.norm(x)).reshape(len(x),4,-1)
        k,v = self.kv(context).reshape(len(context),2,4,-1).unbind(1)
        outputs=[]
        large=sum(c[2].shape[1] for c in edges)*self.width*12>128*1024**2
        for begin,end,index,neighbors,present in edges:
            if neighbors is None:
                args=(q[begin:end],k,v,index,valid[begin:end],context_valid)
                outputs.append(execute(self.attend,*args,large=large))
            else:
                args=(q[begin:end],k,v,neighbors,present,valid[begin:end],context_valid)
                outputs.append(execute(self.padded_attend,*args,large=large))
        x = x+self.out(torch.cat(outputs))
        x = x+self.ff(x)
        return torch.where(valid[:,None],x,0.)

    def padded_attend(self,q,k,v,neighbors,present,valid,context_valid):
        mask=present & context_valid[neighbors] & valid[:,None]
        query=torch.where(valid[:,None,None],q,0.)
        keys=torch.where(mask[...,None,None],k[neighbors],0.)
        values=torch.where(mask[...,None,None],v[neighbors],0.)
        scores=(query[:,None]*keys).sum(-1)/math.sqrt(self.width/4)
        weights=masked_softmax(scores,mask[...,None],1)
        return (weights[...,None]*values).sum(1).flatten(1)


class GlobalBlock(nn.Module):
    """Exact global attention with bounded query chunks, including FP32 execution."""
    def __init__(self,width=128):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,width*3)
        self.out=nn.Linear(width,width)
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))

    def forward(self,x,valid):
        x=torch.where(valid[:,None],x,0.)
        if len(x)<=2048:
            q,k,v=self.qkv(self.norm(x)).reshape(len(x),3,4,-1).permute(1,2,0,3).unbind(0)
            attended=attention(q[None],k[None],v[None],attn_mask=valid[None,None,None,:])[0]
            value=x+self.out(attended.transpose(0,1).flatten(1))
            return torch.where(valid[:,None],value+self.ff(value),0.)
        ids=valid.nonzero().flatten()
        if not len(ids):return x*0
        selected=x[ids]
        q,k,v=self.qkv(self.norm(selected)).reshape(len(ids),3,4,-1).permute(1,2,0,3).unbind(0)
        outputs=[]
        for begin in range(0,len(ids),512):
            args=(q[:,begin:begin+512],k,v)
            outputs.append(execute(attention,*args,large=len(ids)**2*16>64*1024**2))
        value=selected+self.out(torch.cat(outputs,1).transpose(0,1).flatten(1))
        value=value+self.ff(value)
        return torch.zeros_like(x).index_copy(0,ids,value)


class ParentAttention(nn.Module):
    """Exact one-key attention: softmax is one, so Q/K and query normalization are dead."""
    def __init__(self,width,context_width):
        super().__init__()
        original=SparseAttention(width,context_width)
        # Construct the same effective initialization without keeping unused Q/K parameters.
        self.value_weight=nn.Parameter(original.kv.weight[width:].detach().clone())
        self.value_bias=nn.Parameter(original.kv.bias[width:].detach().clone())
        self.out,self.ff=original.out,original.ff

    def forward(self,x,context,edges,valid,context_valid):
        x=torch.where(valid[:,None],x,0.)
        context=torch.where(context_valid[:,None],context,0.)
        values=F.linear(context,self.value_weight,self.value_bias)
        if edges[0][3] is None:
            if any(c[4]!=1 or c[2].shape[1]!=c[1]-c[0] for c in edges):raise ValueError('Unique geometric parent required')
            parents=torch.cat([c[2][1] for c in edges])
            present=context_valid[parents] & valid
        else:
            if any(c[3].shape[1]!=1 for c in edges):raise ValueError('Unique geometric parent required')
            parents=torch.cat([c[3][:,0] for c in edges])
            present=torch.cat([c[4][:,0] for c in edges]) & context_valid[parents] & valid
        result=x+self.out(torch.where(present[:,None],values[parents],0.))
        result=result+self.ff(result)
        return torch.where(valid[:,None],result,0.)


def canonical_weights(state):
    """Map the frozen one-parent QKV state to the equivalent effective V-only state."""
    result=dict(state)
    for encoder in ('online','target'):
        for name,width in (('m_to_f',64),('c_to_m',96)):
            prefix=f'{encoder}.cross.{name}.'
            if prefix+'kv.weight' not in result:continue
            result[prefix+'value_weight']=result[prefix+'kv.weight'][width:].clone()
            result[prefix+'value_bias']=result[prefix+'kv.bias'][width:].clone()
            for tail in ('q.weight','q.bias','kv.weight','kv.bias','norm.weight','norm.bias'):
                result.pop(prefix+tail,None)
    return result


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
        self.cross = nn.ModuleDict({name:(ParentAttention if name in ('m_to_f','c_to_m') else SparseAttention)(WIDTHS[q],WIDTHS[k]) for name,q,k in (
            ('f_to_m',1,0),('m_to_f',0,1),('m_to_c',2,1),('c_to_m',1,2))})
        self.norms = nn.ModuleList(nn.LayerNorm(w) for w in WIDTHS)

    def conditions(self,sample):
        return self.memory([c.physical for c in sample.channels])

    def convolve(self,clean,kernel):
        count,height,width=clean.shape
        return F.conv2d(clean.reshape(1,count,height,width),kernel.reshape(count*32,1,3,3),padding=1,groups=count).reshape(count,32,height,width)

    def native(self, values, keep, kernel, bias, maps):
        if values.ndim==2:
            result=self.native(values[None],keep[None],kernel[None],bias[None],maps)
            return tuple(v[0] for v in result)
        count,height,width=values.shape
        clean = torch.where(keep,values,0.)
        # One convolution per channel/view, reused by all three independent aggregations.
        support = F.conv2d(keep[:,None].float(),values.new_ones(1,1,3,3),padding=1)
        feature=self.convolve(clean,kernel)
        feature=F.gelu(feature*9/support.clamp_min(1)+bias[:,:,None,None])
        feature=feature.permute(0,2,3,1).reshape(count,-1,32)
        feature=torch.where(keep.flatten(1)[...,None],feature,0.)
        return tuple(value for mapping in maps for value in mapping.aggregate(feature,keep))

    def fuse(self,summary,support,condition,level):
        h=self.projection[level](torch.cat((summary,support[...,None]),-1))
        content=torch.cat((h,condition[:,None].expand(-1,h.shape[1],-1)),-1)
        present=support>0
        score=self.fusion_key[level](content).squeeze(-1)
        weights=masked_softmax(score,present,0)
        return (self.fusion_value[level](content)*weights[...,None]).sum(0)

    def forward(self, sample, layout, keeps):
        geo,parts,maps,edges,relations = layout
        conditions = self.conditions(sample)
        kernels,biases = self.convolution(conditions['conv'])
        summaries = [[],[],[]]
        supports = [[],[],[]]
        groups={}
        active=torch.stack([keep.any() for keep in keeps]).tolist()
        order=[]
        for i,c in enumerate(sample.channels):
            if active[i]:groups.setdefault(c.grid,[]).append(i)
        if not groups:raise ValueError('NO_VISIBLE_CONTEXT')
        for grid,indices in groups.items():
            capacity=max(1,262144//(grid.shape[0]*grid.shape[1]))
            for begin in range(0,len(indices),capacity):
                selected=indices[begin:begin+capacity]
                def front(values,keep,kernel,bias,mapping=maps[selected[0]]):
                    return self.native(values,keep,kernel,bias,mapping)
                args=(torch.stack([sample.channels[i].values for i in selected]),torch.stack([keeps[i] for i in selected]),kernels[selected],biases[selected])
                values=execute(front,*args,large=args[0].numel()*128>128*1024**2)
                for level in range(3):
                    summaries[level].extend(values[level*2].unbind(0))
                    supports[level].extend(values[level*2+1].unbind(0))
                order.extend(selected)
        features,valids,metadata = [],[],[]
        for level,part in enumerate(parts):
            support = torch.stack(supports[level])
            chunks=[]
            for start in range(0,len(part),4096):
                summary=torch.stack([v[start:start+4096] for v in summaries[level]])
                def fuse(a,b,c,level=level):return self.fuse(a,b,c,level)
                args=(summary,support[:,start:start+4096],conditions['fusion'][order])
                chunks.append(execute(fuse,*args,large=len(part)*len(order)*WIDTHS[level]*12>128*1024**2))
            x=torch.cat(chunks)
            # Validity is support union (any), never a sum of overlapping channel areas.
            valid = (support>0).any(0)
            centers = torch.tensor(part.centers,device=x.device)+torch.tensor(geo.origin,device=x.device,dtype=torch.float64)
            x = x+self.position[level](position_features(centers,geo))
            features.append(torch.where(valid[:,None],x,0.));valids.append(valid)
            metadata.append(ScaleTokens(features[-1],centers,support,valid,
                relations[level][2] if level<2 else None,relations[level-1][1] if level>0 else None,
                (geo,part),tuple(order)))
        for level in range(2):
            for block in self.branches[level]:
                args=(features[level],features[level],edges[level],valids[level],valids[level])
                features[level] = execute(block,*args,large=features[level].numel()*32>64*1024**2)
        if not bool(valids[2].any()):
            raise ValueError('NO_VISIBLE_CONTEXT')
        for block in self.coarse:
            features[2] = execute(block,features[2],valids[2],large=features[2].numel()*32>64*1024**2)
        # One bottom-up and then top-down sweep after the independent branch stacks.
        for name,q,k,edge in (
            ('f_to_m',1,0,relations[0][1]),('m_to_c',2,1,relations[1][1]),
            ('c_to_m',1,2,relations[1][0]),('m_to_f',0,1,relations[0][0])):
            block=self.cross[name];args=(features[q],features[k],edge,valids[q],valids[k])
            features[q] = execute(block,*args,large=features[q].numel()*32>64*1024**2)
        features=[norm(x)*valid[:,None] for norm,x,valid in zip(self.norms,features,valids)]
        for token,x in zip(metadata,features):token.features=x
        return EncoderOutput(features,valids,conditions,metadata)


class HierarchicalMAE(nn.Module):
    def __init__(self, *, patch_metres, mask_ratio=.55, encoder_depth=2, lambda_jepa=0., ema_momentum=.996,
                 frontend='physical_descriptor',channel_keys=None):
        super().__init__()
        if not 0<mask_ratio<1 or not 0<=ema_momentum<1 or lambda_jepa<0:
            raise ValueError('Invalid masking, EMA or JEPA weight')
        self.patch_metres,self.mask_ratio = patch_metres,mask_ratio
        self.lambda_jepa,self.ema_momentum = lambda_jepa,ema_momentum
        if frontend=='physical_descriptor':self.online = Encoder(encoder_depth)
        elif frontend=='shared_static_band_embedding':
            from .descriptor_free import DescriptorFreeEncoder
            self.online = DescriptorFreeEncoder(encoder_depth,channel_keys)
        else:raise ValueError('Unknown frontend')
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
        geo=metric_geometry(sample.channels,self.patch_metres)
        relative=tuple((g.shape,g.gsd_xy,g.affine[2]-geo.origin[0],g.affine[5]-geo.origin[1]) for g in grids)
        key=(relative,geo.extent,self.patch_metres,str(device))
        if key not in self._layouts:
            if sum(v[1] for v in self._layouts.values())>256*1024**2 or sum(g.shape[0]*g.shape[1] for g in set(grids))>1000000:
                self._layouts.clear()
            _,parts,gsd=partitions(grids,self.patch_metres)
            unique={g:tuple(area_map(g,geo,p,device) for p in parts) for g in set(grids)}
            maps=[unique[g] for g in grids]
            edges=[neighborhood(parts[0],(gsd[0]*3,gsd[1]*3),device),
                   neighborhood(parts[1],(self.patch_metres,self.patch_metres),device)]
            import numpy as np
            relations=[]
            for level,p in enumerate(parts[:2]):
                index=np.stack((np.arange(len(p)),p.parent))
                relations.append((sparse_edges(index,len(p),device),sparse_edges(index[::-1],len(parts[level+1]),device),torch.tensor(p.parent,device=device)))
            layout=(geo,parts,maps,edges,relations)
            tensors={}
            def collect(value):
                if isinstance(value,torch.Tensor) and value.device.type!='cpu':tensors[value.data_ptr()]=value
                elif isinstance(value,(tuple,list)):
                    for v in value:collect(v)
                elif hasattr(value,'source'):
                    for v in vars(value).values():collect(v)
            collect((maps,edges,relations))
            cost=sum(t.numel()*t.element_size() for t in tensors.values())
            while self._layouts and (len(self._layouts)>=32 or sum(v[1] for v in self._layouts.values())+cost>256*1024**2):
                self._layouts.popitem(last=False)
            self._layouts[key]=(layout,cost)
        self._layouts.move_to_end(key)
        cached=self._layouts[key][0]
        return (geo,*cached[1:])

    def available_tasks(self,sample):
        layout=self.layout(sample)
        occupied=torch.zeros(len(layout[1][2]),device=sample.channels[0].values.device,dtype=torch.bool)
        for c,maps in zip(sample.channels,layout[2]):
            support=maps[2].support(c.valid)
            occupied |= support>0
        valid_count=int(torch.stack([c.valid.any() for c in sample.channels]).sum())
        kinds=(['A'] if int(occupied.sum())>=2 else [])+(['B1'] if valid_count>=2 else [])
        if not kinds:
            raise ValueError('No valid reconstruction context')
        return kinds

    def masks(self,sample,layout,kind,seed,epoch):
        geo,parts,maps,_,_=layout
        device=sample.channels[0].values.device
        patch_mask=torch.zeros(len(parts[2]),device=device,dtype=torch.bool)
        eligible=[i for i,present in enumerate(torch.stack([c.valid.any() for c in sample.channels]).tolist()) if present]
        hidden_bands=set()
        if kind=='A':
            occupied=torch.zeros_like(patch_mask)
            for c,mapping in zip(sample.channels,maps):
                occupied |= mapping[2].support(c.valid)>0
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
                decoded[level]=execute(block,x,valid,large=x.numel()*32>64*1024**2)
            else:
                block=self.decoder[level];args=(x,x,edges[level],valid,valid)
                decoded[level]=execute(block,*args,large=x.numel()*32>64*1024**2)
        return decoded

    def readout(self,features,condition):
        def project(values,physical):
            context=physical.expand(len(values),-1)
            return self.output(torch.cat((values,context),-1)).squeeze(-1)
        if len(features)<=100000:return project(features,condition)
        # Bound output-head activations without changing any native targets.
        return torch.cat([execute(project,chunk,condition,large=True) for chunk in features.split(8192)])

    def forward(self,samples,*,kind='A',seed=17,epoch=0,return_predictions=False,return_objectives=False):
        losses,details=[],[]
        for sample in samples:
            layout=self.layout(sample)
            keeps,targets,patch_mask=self.masks(sample,layout,kind,seed,epoch)
            encoded,valids,conditions=self.online(sample,layout,keeps)
            query_valid=[]
            for level,part in enumerate(layout[1]):
                supported=torch.zeros(len(part),device=encoded[0].device,dtype=torch.bool)
                for c,mapping in zip(sample.channels,layout[2]):
                    supported |= mapping[level].support(c.valid)>0
                query_valid.append(supported)
            decoded=self.decode(encoded,valids,layout,query_valid)
            channel_losses,predictions=[],[]
            target_flags=torch.stack([mask.any() for mask in targets]).tolist()
            for i,c in enumerate(sample.channels):
                if not target_flags[i] and not return_predictions:
                    predictions.append(None)
                    continue
                field=self.readout(decoded[0],conditions['output'][i])
                predicted=layout[2][i][0].native(field).reshape(c.values.shape)
                mask=targets[i]
                if target_flags[i]:
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
            scalars=torch.stack((mae.detach(),jepa.detach())).cpu().tolist()
            visible_counts=torch.stack([v.sum() for v in valids]).tolist()
            target_count=int(torch.stack([t.sum() for t in targets]).sum())
            details.append(dict(id=sample.identifier,task=kind,mae=scalars[0],jepa=scalars[1],
                jepa_scales=[float(v.detach()) for v in scale_losses],jepa_counts=scale_counts,target_std=target_variance,
                tokens=[len(p) for p in layout[1]],visible_tokens=visible_counts,
                attention_edges=[sum(chunk[2].shape[1] for chunk in e) for e in layout[3]],targets=target_count,
                predictions=predictions if return_predictions else None,target_masks=targets if return_predictions else None,
                online_features=encoded if return_predictions else None))
            if return_objectives:
                details[-1]['objective_tensors']={'mae':mae,'jepa':jepa,**{f'jepa_scale_{i}':v for i,v in enumerate(scale_losses)}}
        return torch.stack(losses),details
