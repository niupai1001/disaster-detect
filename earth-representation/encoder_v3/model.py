"""Native-grid V3 reference path; geometry never reads reflectance values."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class BandObservation:
    reflectance: torch.Tensor
    valid_mask: torch.Tensor
    wavelengths_nm: torch.Tensor
    response: torch.Tensor
    affine: tuple
    crs: str
    record_id: str = ''
    visible: torch.Tensor | None = None

    def validate(self):
        if self.reflectance.ndim != 2 or self.valid_mask.shape != self.reflectance.shape:
            raise ValueError('Expected reflectance and valid_mask [H,W]')
        if self.valid_mask.dtype != torch.bool or not self.crs:
            raise ValueError('Boolean mask and explicit CRS required')
        a,b,_,d,e,_ = self.affine
        if not self.reflectance.is_floating_point() or min(self.reflectance.shape)<1 or not all(math.isfinite(float(x)) for x in self.affine):
            raise ValueError('Floating reflectance, nonempty grid and finite affine required')
        # shortcut: reference geometry supports projected north-up grids only;
        # rotated/geographic grids require polygon geometry before admission.
        if b or d or a <= 0 or e >= 0:
            raise ValueError('Reference path requires north-up projected metric grids')
        from rasterio.crs import CRS
        crs = CRS.from_user_input(self.crs)
        if not crs.is_projected or crs.linear_units not in ('metre', 'meter'):
            raise ValueError('Metric projected CRS required; never resample silently')
        if self.visible is not None and (self.visible.shape != self.reflectance.shape or self.visible.dtype != torch.bool):
            raise ValueError('Visibility must be boolean [H,W]')

    def keep(self):
        self.validate()
        return self.valid_mask if self.visible is None else self.valid_mask & self.visible

    def cells(self):
        h,w = self.reflectance.shape
        a,_,c,_,e,f = self.affine
        y,x = torch.meshgrid(torch.arange(h,device=self.reflectance.device),torch.arange(w,device=self.reflectance.device),indexing='ij')
        x,y = x.flatten().double(),y.flatten().double()
        return torch.stack((c+x*a,f+(y+1)*e,c+(x+1)*a,f+y*e),-1)


@dataclass
class Representation:
    tokens: torch.Tensor
    cells: list  # Exact represented native cell unions, not bounding-box areas.
    centers: torch.Tensor
    bounds: torch.Tensor
    valid: torch.Tensor
    crs: str
    receptive_cells: list | None = None  # Local convolution/Merge support before backbone.
    spatial_index: object | None = None
    gsd_xy: torch.Tensor | None = None
    support_xy: torch.Tensor | None = None
    srf_condition: torch.Tensor | None = None
    source_bands: tuple | None = None  # Audit identifiers only; never embedded.
    merge_scale_m: float | None = None
    origin_xy: tuple | None = None
    full_input_cells: tuple | None = None  # Shared union for every Z token after full attention; regions may overlap.


def positional(xy, width=128):
    freq = torch.exp(torch.arange(width//4,device=xy.device,dtype=xy.dtype)*(-math.log(10000)/(width//4)))
    phase = xy[...,None]*freq
    return torch.cat((phase.sin().flatten(-2),phase.cos().flatten(-2)),-1)


class SRFEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave = nn.Sequential(nn.Linear(1,64),nn.GELU(),nn.Linear(64,32))

    def forward(self, wavelength, response):
        if wavelength.ndim != 1 or wavelength.shape != response.shape or len(wavelength)<2:
            raise ValueError('SRF requires matching [M], M>=2')
        if not bool(torch.isfinite(wavelength).all() & torch.isfinite(response).all()):
            raise ValueError('Nonfinite SRF')
        if not bool(((wavelength[1:]-wavelength[:-1])>0).all() & (response>=0).all() & (wavelength>0).all()):
            raise ValueError('SRF wavelengths must increase; response must be nonnegative')
        area = torch.trapezoid(response,wavelength)
        if not bool((area>0) & torch.isfinite(area)):
            raise ValueError('Zero SRF area')
        features = self.wave((wavelength/1000)[:,None])
        return torch.trapezoid(features*response[:,None],wavelength,dim=0)/area


class Encoder(nn.Module):
    def __init__(self, *, hyper_hidden, local_batch_size=1, native_mode='serial', tile_rows=None, layout_mode='reference'):
        super().__init__()
        if hyper_hidden is None:
            raise ValueError('Hypernetwork hidden width requires explicit candidate configuration')
        if local_batch_size < 1:
            raise ValueError('Positive execution batch size required')
        self.local_batch_size=local_batch_size
        if native_mode not in ('serial','grouped') or (tile_rows is not None and tile_rows<1):
            raise ValueError('Invalid native execution configuration')
        if native_mode=='grouped' and tile_rows is not None:
            raise ValueError('Grouped tiling is not yet an accepted execution path')
        self.native_mode=native_mode;self.tile_rows=tile_rows
        if layout_mode not in ('reference','indexed'):raise ValueError('Unknown layout execution mode')
        self.layout_mode=layout_mode
        self.srf = SRFEncoder()
        self.hyper = nn.Sequential(nn.Linear(34,hyper_hidden),nn.GELU(),nn.Linear(hyper_hidden,320))
        self.project = nn.Linear(32,128)
        self.cls = nn.Parameter(torch.zeros(1,1,128))
        self.local = nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        self.global_net = nn.TransformerEncoder(nn.TransformerEncoderLayer(128,4,512,dropout=0.,activation='gelu',batch_first=True,norm_first=True),4,enable_nested_tensor=False)

    def native(self, band):
        keep = band.keep()
        if not bool(torch.isfinite(band.reflectance[keep]).all()):
            raise ValueError('Nonfinite visible reflectance')
        a,_,_,_,e,_ = band.affine
        condition = self.srf(band.wavelengths_nm,band.response)
        params = self.hyper(torch.cat((condition,condition.new_tensor([a,-e]).log())))
        values = torch.where(keep,band.reflectance,0)[None,None]
        outputs=[];height=values.shape[-2];rows=self.tile_rows or height
        for begin in range(0,height,rows):
            end=min(begin+rows,height);lo=max(0,begin-1);hi=min(height,end+1)
            count=F.conv2d(keep[None,None,lo:hi].to(values.dtype),values.new_ones(1,1,3,3),padding=1)
            output=F.conv2d(values[:,:,lo:hi],params[:288].reshape(32,1,3,3),padding=1)
            output=output*(9/count.clamp_min(1))+params[288:][None,:,None,None]
            outputs.append(output[:,:,begin-lo:end-lo])
        features=torch.cat(outputs,dim=2)
        features = F.gelu(features)[0].permute(1,2,0).reshape(-1,32)
        return features, keep.flatten(), condition

    def native_grouped(self, bands):
        buckets={}
        for i,band in enumerate(bands):
            band.validate()
            key=(tuple(band.reflectance.shape),band.reflectance.device,band.reflectance.dtype)
            buckets.setdefault(key,[]).append(i)
        result=[None]*len(bands)
        for indices in buckets.values():
            batch=[bands[i] for i in indices]; masks=torch.stack([b.keep() for b in batch])
            raw=torch.stack([b.reflectance for b in batch])
            if not bool(torch.isfinite(raw[masks]).all()): raise ValueError('Nonfinite visible reflectance')
            conditions=torch.stack([self.srf(b.wavelengths_nm,b.response) for b in batch])
            gsd=conditions.new_tensor([[b.affine[0],-b.affine[4]] for b in batch]).log()
            params=self.hyper(torch.cat((conditions,gsd),-1))
            values=torch.where(masks,raw,0);count=F.conv2d(masks[:,None].to(values.dtype),values.new_ones(1,1,3,3),padding=1)
            h,w=values.shape[-2:];size=len(batch)
            feature=F.conv2d(values[None],params[:,:288].reshape(size*32,1,3,3),padding=1,groups=size).reshape(size,32,h,w)
            feature=F.gelu(feature*(9/count.clamp_min(1))+params[:,288:,None,None])
            for j,index in enumerate(indices): result[index]=(feature[j].permute(1,2,0).reshape(-1,32),masks[j].flatten(),conditions[j])
        return result

    def _prepare(self, bands, *, patch_m, origin_xy, native=None):
        if not math.isfinite(patch_m) or patch_m<=0 or not bands or len(origin_xy)!=2 or not all(math.isfinite(x) for x in origin_xy):
            raise ValueError('Positive explicit patch scale and visible bands required')
        if len({b.crs for b in bands}) != 1:
            raise ValueError('Transform geometry to common metric CRS before encoding')
        groups=[]; represented=[]; receptive=[];metadata_centers=[];metadata_bounds=[]
        gsd=[];conditions=[];source_bands=[]
        def physical(count,band,condition):
            if count:
                gsd.append(condition.new_tensor([band.affine[0],-band.affine[4]]).expand(count,-1))
                conditions.append(condition.expand(count,-1));source_bands.extend([band.record_id]*count)
        if native is None:native=self.native_grouped(bands) if self.native_mode=='grouped' else [self.native(band) for band in bands]
        for band,(features,keep,condition) in zip(bands,native,strict=True):
            before=len(groups)
            if self.layout_mode=='indexed':
                from .layout import native_layout
                layout=native_layout(band,patch_m,origin_xy)
                if layout is None:continue
                index=torch.as_tensor(layout['indices'],device=features.device)
                xy=torch.as_tensor(layout['xy'],device=features.device,dtype=features.dtype)
                combined=self.project(features[index])+positional(xy)
                groups.extend(combined.split(layout['lengths']))
                represented.extend(torch.as_tensor(layout['represented'],device=features.device).split(layout['represented_lengths']))
                receptive.extend(torch.as_tensor(layout['receptive'],device=features.device).split(layout['receptive_lengths']))
                metadata_centers.extend(torch.as_tensor(layout['centers'],device=features.device).unbind())
                metadata_bounds.extend(torch.as_tensor(layout['bounds'],device=features.device).unbind())
                physical(len(groups)-before,band,condition)
                continue
            cells=band.cells(); centers=(cells[:,:2]+cells[:,2:])/2
            origin=centers.new_tensor(origin_xy)
            ids=torch.floor((centers-origin)/patch_m).long()
            for key in torch.unique(ids,dim=0):
                belongs=(ids==key).all(1)
                selected=belongs & keep
                if not bool(selected.any()):
                    continue
                xy=(centers[selected]-origin).to(features.dtype)
                groups.append(self.project(features[selected])+positional(xy))
                # Assignment geometry is independent of task visibility and values.
                represented.append(cells[belongs & band.valid_mask.flatten()])
                neighborhood=F.max_pool2d(selected.reshape(band.reflectance.shape)[None,None].float(),3,stride=1,padding=1)[0,0].bool().flatten()
                receptive.append(cells[neighborhood & keep])
                region=represented[-1]
                metadata_centers.append(((region[:,:2]+region[:,2:])/2).mean(0))
                metadata_bounds.append(torch.cat((region[:,:2].amin(0),region[:,2:].amax(0))))
            physical(len(groups)-before,band,condition)
        if not groups:
            raise ValueError('No visible tokens')
        bounds=torch.stack(metadata_bounds)
        metadata=dict(gsd_xy=torch.cat(gsd),support_xy=bounds[:,2:]-bounds[:,:2],srf_condition=torch.cat(conditions),source_bands=tuple(source_bands),merge_scale_m=patch_m,origin_xy=tuple(origin_xy),full_input_cells=tuple(receptive))
        return groups,represented,receptive,torch.stack(metadata_centers),bounds,metadata

    def _local_tokens(self, groups):
        # Bucket by actual length, never round GSD or truncate local positions.
        buckets={}
        for index,group in enumerate(groups):
            buckets.setdefault(len(group),[]).append(index)
        tokens=[None]*len(groups)
        for indices in buckets.values():
            for start in range(0,len(indices),self.local_batch_size):
                chosen=indices[start:start+self.local_batch_size]
                batch=torch.stack([groups[i] for i in chosen])
                outputs=self.local(torch.cat((self.cls.expand(len(chosen),-1,-1),batch),1))[:,0]
                for i,value in zip(chosen,outputs): tokens[i]=value
        return torch.stack(tokens)

    def forward(self, bands, *, patch_m, origin_xy):
        groups,represented,receptive,centers,bounds,metadata=self._prepare(bands,patch_m=patch_m,origin_xy=origin_xy)
        z=self._local_tokens(groups)
        z=self.global_net((z+positional((centers-centers.new_tensor(origin_xy)).to(z.dtype)))[None])[0]
        return Representation(z,represented,centers,bounds,torch.ones(len(z),dtype=torch.bool,device=z.device),bands[0].crs,receptive,**metadata)

    def forward_batch(self, observations):
        """Independent observations, grouped native/local work and padded global attention.

        Each item is (visible bands, physical patch size, physical origin).
        The batch dimension and padding mask forbid cross-observation attention.
        """
        if not observations:raise ValueError('Nonempty observation microbatch required')
        bands=[band for visible,_,_ in observations for band in visible]
        native=self.native_grouped(bands) if self.native_mode=='grouped' else [self.native(band) for band in bands]
        prepared=[];offset=0
        for visible,patch,origin in observations:
            prepared.append(self._prepare(visible,patch_m=patch,origin_xy=origin,native=native[offset:offset+len(visible)]))
            offset+=len(visible)
        lengths=[len(item[0]) for item in prepared]
        tokens=self._local_tokens([group for item in prepared for group in item[0]]).split(lengths)
        positioned=[z+positional((item[3]-item[3].new_tensor(origin)).to(z.dtype)) for z,item,(_,_,origin) in zip(tokens,prepared,observations)]
        padded=nn.utils.rnn.pad_sequence(positioned,batch_first=True)
        mask=torch.arange(padded.shape[1],device=padded.device)[None]>=torch.tensor(lengths,device=padded.device)[:,None]
        output=self.global_net(padded,src_key_padding_mask=mask)
        result=[]
        for index,(length,item,(visible,_,_)) in enumerate(zip(lengths,prepared,observations)):
            _,represented,receptive,centers,bounds,metadata=item;z=output[index,:length]
            result.append(Representation(z,represented,centers,bounds,torch.ones(length,dtype=torch.bool,device=z.device),visible[0].crs,receptive,**metadata))
        return result


def union_overlap(cells, targets):
    low=torch.maximum(targets[:,None,:2],cells[None,:,:2])
    high=torch.minimum(targets[:,None,2:],cells[None,:,2:])
    return (high-low).clamp_min(0).prod(-1).sum(-1)


class ReflectanceDecoder(nn.Module):
    def __init__(self, *, indexed=False, cell_indexed=False):
        super().__init__()
        self.indexed=indexed
        if cell_indexed and not indexed:raise ValueError('Cell lookup requires indexed Decoder')
        self.cell_indexed=cell_indexed
        self.mlp=nn.Sequential(nn.Linear(162,128),nn.GELU(),nn.Linear(128,1))

    def forward(self, representation, targets, srf_condition, *, kind, mask_bounds=None):
        if kind not in ('A','B1','B2') or targets.ndim!=2 or targets.shape[1]!=4:
            raise ValueError('Explicit task and target rectangles [Q,4] required')
        if srf_condition.shape!=(32,) or not bool((targets[:,2:]>targets[:,:2]).all()):
            raise ValueError('Invalid SRF condition or target support')
        if self.indexed:
            from .spatial_index import aggregate
            feature,position,supported=aggregate(representation,targets,kind=kind,mask_bounds=mask_bounds,cell_indexed=self.cell_indexed)
            output=self.mlp(torch.cat((feature,position,srf_condition.expand(len(targets),-1)),-1))[:,0]
            return torch.where(supported,output,torch.nan),supported
        r=representation; centers=(targets[:,:2]+targets[:,2:])/2
        scale=(r.bounds[:,2:]-r.bounds[:,:2]).clamp_min(torch.finfo(r.bounds.dtype).eps)
        relative=(centers[:,None]-r.centers[None])/scale[None]
        if kind=='A':
            if mask_bounds is None:
                raise ValueError('A requires explicit masked-region geometry')
            m=mask_bounds
            touch=[]
            for cells in r.cells:
                intersects=(cells[:,0]<=m[2])&(cells[:,2]>=m[0])&(cells[:,1]<=m[3])&(cells[:,3]>=m[1])
                strictly_inside=(cells[:,:2]>m[:2]).all(1)&(cells[:,2:]<m[2:]).all(1)
                touch.append((intersects & ~strictly_inside).any())
            touch=torch.stack(touch)
            outside=~((r.centers[:,0]>m[0])&(r.centers[:,0]<m[2])&(r.centers[:,1]>m[1])&(r.centers[:,1]<m[3]))
            weights=(1/(1+relative.square().sum(-1)))*(touch & outside & r.valid)[None]
        else:
            weights=torch.stack([union_overlap(c,targets) for c in r.cells],1)*r.valid[None]
        supported=weights.sum(1)>0
        weights=weights/weights.sum(1,keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        feature=weights.to(r.tokens.dtype)@r.tokens
        position=(weights[...,None]*relative).sum(1).to(feature.dtype)
        output=self.mlp(torch.cat((feature,position,srf_condition.expand(len(targets),-1)),-1))[:,0]
        return torch.where(supported,output,torch.nan),supported


def spatial_hide(band, rectangle):
    cells=band.cells(); m=cells.new_tensor(rectangle)
    intersects=(torch.minimum(cells[:,2:],m[2:])-torch.maximum(cells[:,:2],m[:2])).gt(0).all(1)
    contained=(cells[:,:2]>=m[:2]).all(1)&(cells[:,2:]<=m[2:]).all(1)
    prior=band.keep()
    hidden=BandObservation(band.reflectance,band.valid_mask,band.wavelengths_nm,band.response,band.affine,band.crs,band.record_id,prior & ~intersects.reshape(prior.shape))
    return hidden,contained.reshape(prior.shape)&band.valid_mask
