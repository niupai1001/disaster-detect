"""Value-independent native layout: batch index transfers instead of GPU scalar loops."""
import numpy as np


def native_layout(band,patch_m,origin_xy):
    h,w=band.reflectance.shape;a,_,c,_,e,f=band.affine
    row,col=np.meshgrid(np.arange(h),np.arange(w),indexing='ij');row=row.ravel();col=col.ravel()
    cells=np.stack((c+col*a,f+(row+1)*e,c+(col+1)*a,f+row*e),-1).astype(np.float64)
    centers=(cells[:,:2]+cells[:,2:])/2
    ids=np.floor((centers-np.asarray(origin_xy))/patch_m).astype(np.int64)
    _,inverse=np.unique(ids,axis=0,return_inverse=True)
    order=np.argsort(inverse,kind='stable');cuts=np.flatnonzero(np.diff(inverse[order]))+1
    qa=band.valid_mask.detach().cpu().numpy().ravel()
    visible=qa if band.visible is None else qa & band.visible.detach().cpu().numpy().ravel()
    selected=[];represented=[];receptive=[];positions=[];bounds=[]
    for members in np.split(order,cuts):
        chosen=members[visible[members]]
        if not len(chosen):continue
        selected.append(chosen);region=cells[members[qa[members]]];represented.append(region)
        positions.append(((region[:,:2]+region[:,2:])/2).mean(0))
        bounds.append(np.concatenate((region[:,:2].min(0),region[:,2:].max(0))))
        neighbors=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                yy=row[chosen]+dy;xx=col[chosen]+dx;valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
                neighbors.append(yy[valid]*w+xx[valid])
        neighbors=np.unique(np.concatenate(neighbors));receptive.append(cells[neighbors[visible[neighbors]]])
    if not selected:return None
    return dict(indices=np.concatenate(selected),lengths=[len(x) for x in selected],xy=centers[np.concatenate(selected)]-np.asarray(origin_xy),represented=np.concatenate(represented),represented_lengths=[len(x) for x in represented],receptive=np.concatenate(receptive),receptive_lengths=[len(x) for x in receptive],centers=np.asarray(positions),bounds=np.asarray(bounds))
