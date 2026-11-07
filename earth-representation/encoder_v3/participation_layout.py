"""Approved mechanism B: participation-derived windows and complete-group selection."""
from dataclasses import dataclass,asdict
import hashlib,json,math
import numpy as np


@dataclass(frozen=True)
class AreaEquivalentHillRate:
    """Candidate curve only. All coefficients explicit; area-GSD rule not frozen."""
    amplitude: float
    scale_m: float
    power: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.amplitude,self.scale_m,self.power)) or not 0<self.amplitude<=1 or self.scale_m<=0 or self.power<=0:raise ValueError('Explicit valid participation curve coefficients required')

    def __call__(self,gx,gy):
        return self.amplitude/(1+(math.sqrt(gx*gy)/self.scale_m)**self.power)

    def get_config(self):return dict(family='candidate_area_equivalent_hill',effective_gsd='sqrt(gx*gy)',**asdict(self))


def participation_layout(band,*,rate_curve,selection_seed,origin_xy):
    band.validate()
    if type(selection_seed) is not int or len(origin_xy)!=2 or not all(math.isfinite(x) for x in origin_xy):raise ValueError('Explicit seed and finite physical origin required')
    gx,_,left,_,e,top=band.affine;gy=-e;rate=float(rate_curve(gx,gy))
    if not math.isfinite(rate) or not 0<=rate<1:raise ValueError('Participation rate must be finite in [0,1)')
    span=1/(1-rate);window=np.asarray((gx*span,gy*span));h,w=band.reflectance.shape
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');yy=yy.ravel();xx=xx.ravel()
    cells=np.stack((left+xx*gx,top+(yy+1)*e,left+(xx+1)*gx,top+yy*e),-1).astype(np.float64);centers=(cells[:,:2]+cells[:,2:])/2
    qa=band.valid_mask.detach().cpu().numpy().ravel();keep=band.keep().detach().cpu().numpy().ravel();n=int(keep.sum())
    if not n:return None
    ids=np.floor((centers-np.asarray(origin_xy))/window).astype(np.int64);keys,inverse=np.unique(ids,axis=0,return_inverse=True)
    order=np.argsort(inverse,kind='stable');groups=np.split(order,np.flatnonzero(np.diff(inverse[order]))+1);sizes=np.bincount(inverse[keep],minlength=len(keys));eligible=np.flatnonzero(sizes>=2)
    prefix=json.dumps([selection_seed,band.crs,list(origin_xy),window.tolist()],separators=(',',':')).encode()
    candidates=sorted(eligible,key=lambda i:hashlib.sha256(prefix+keys[i].tobytes()).digest());cumulative=np.r_[0,np.cumsum(sizes[candidates])];count=int(np.argmin(abs(cumulative-rate*n)))
    merged=np.zeros(len(keys),dtype=bool);merged[candidates[:count]]=True
    output=[];group_keys=[];token_window=[]
    for i,members in enumerate(groups):
        if not sizes[i]:continue
        if merged[i]:output.append(members);group_keys.append(('merge',int(keys[i,0]),int(keys[i,1])));token_window.append(window)
        else:
            for pixel in members[keep[members]]:output.append(np.array([pixel]));group_keys.append(('native',int(yy[pixel]),int(xx[pixel])));token_window.append((gx,gy))
    chosen=[members[keep[members]] for members in output];represented=[cells[members[qa[members]]] for members in output];coverage=[cells[members] for members in output];visible=[cells[members] for members in chosen]
    indices=np.concatenate(chosen);lengths=[len(x) for x in chosen];offsets=np.r_[0,np.cumsum(lengths)];represented_lengths=[len(x) for x in represented];region=np.concatenate(represented);region_start=np.r_[0,np.cumsum(represented_lengths)[:-1]]
    # Batch native 3x3 support construction; no nine Python launches per singleton.
    dy,dx=np.meshgrid(np.arange(-1,2),np.arange(-1,2),indexing='ij');ny=yy[indices,None]+dy.ravel();nx=xx[indices,None]+dx.ravel();neighbor=ny*w+nx;inside=(ny>=0)&(ny<h)&(nx>=0)&(nx<w);neighbor=np.clip(neighbor,0,h*w-1);valid=inside&keep[neighbor]
    receptive=[]
    for lo,hi in zip(offsets[:-1],offsets[1:]):
        positions=neighbor[lo:hi][valid[lo:hi]]
        if hi-lo>1:positions=np.unique(positions)
        receptive.append(cells[positions])
    boxes=np.c_[np.minimum.reduceat(region[:,:2],region_start,axis=0),np.maximum.reduceat(region[:,2:],region_start,axis=0)]
    center=np.add.reduceat((region[:,:2]+region[:,2:])/2,region_start,axis=0)/np.asarray(represented_lengths)[:,None]
    participants=int(sizes[merged].sum());audit=dict(target_rate=rate,actual_rate=participants/n,visible_positions=n,merged_participants=participants,independent_positions=n-participants,merged_groups=int(merged.sum()),tokens=len(output),candidate_span_pixels=span,candidate_window_m=window.tolist(),eligible_participant_capacity=int(sizes[eligible].sum()),selection_rule='nearest prefix in deterministic physical-hash ordering',selection_seed=selection_seed,group_size_min=int(sizes[merged].min()) if merged.any() else None,group_size_max=int(sizes[merged].max()) if merged.any() else None)
    return dict(indices=indices,lengths=lengths,xy=centers[indices]-np.asarray(origin_xy),represented=region,represented_lengths=represented_lengths,receptive=np.concatenate(receptive),receptive_lengths=[len(x) for x in receptive],centers=center,bounds=boxes,coverage=coverage,visible=visible,group_ids=group_keys,window_m=window,token_window_m=np.asarray(token_window),participation_audit=audit)
