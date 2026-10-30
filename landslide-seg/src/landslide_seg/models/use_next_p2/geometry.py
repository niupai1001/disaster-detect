"""Same exact polygon intersections as P0, batched through GEOS instead of Python."""
import math

import numpy as np
from pyproj import CRS,Transformer
import shapely
import torch


def area_grid(asset,grid=None):
    asset.validate();_,h,w=asset.image.shape
    a,b,c,d,e,f=asset.sample_grid['transform'];source=CRS(asset.sample_grid['crs'])
    lon,lat=Transformer.from_crs(source,4326,always_xy=True).transform(a*w/2+b*h/2+c,d*w/2+e*h/2+f)
    target=CRS(grid['crs']) if grid else CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m')
    yy,xx=np.mgrid[:h+1,:w+1]
    x,y=Transformer.from_crs(source,target,always_xy=True).transform(a*xx+b*yy+c,d*xx+e*yy+f)
    x0,y0=math.floor(x.min()/120)*120,math.floor(y.min()/120)*120
    nh,nw=math.ceil((y.max()-y0)/120)*4,math.ceil((x.max()-x0)/120)*4
    if grid:
        x0,y0=grid['origin'];nh,nw=grid['shape']
        if nh%4 or nw%4:raise ValueError('Shared grid must contain complete120m cells')
    vertices=np.stack((np.stack((x[:-1,:-1],y[:-1,:-1]),-1),np.stack((x[:-1,1:],y[:-1,1:]),-1),
                       np.stack((x[1:,1:],y[1:,1:]),-1),np.stack((x[1:,:-1],y[1:,:-1]),-1)),-2).reshape(-1,4,2)
    polygons=shapely.polygons(vertices);bounds=shapely.bounds(polygons)
    left=np.maximum(0,np.floor((bounds[:,0]-x0)/30).astype(int));right=np.minimum(nw,np.ceil((bounds[:,2]-x0)/30).astype(int))
    bottom=np.maximum(0,np.floor((bounds[:,1]-y0)/30).astype(int));top=np.minimum(nh,np.ceil((bounds[:,3]-y0)/30).astype(int))
    rows=[];cols=[];values=[]
    for dx in range(max(0,int(np.max(right-left)))):
        for dy in range(max(0,int(np.max(top-bottom)))):
            u,v=left+dx,bottom+dy;ids=np.flatnonzero((u<right)&(v<top))
            cells=shapely.box(x0+u[ids]*30,y0+v[ids]*30,x0+(u[ids]+1)*30,y0+(v[ids]+1)*30)
            areas=shapely.area(shapely.intersection(polygons[ids],cells));keep=areas>1e-8
            rows.append(v[ids[keep]]*nw+u[ids[keep]]);cols.append(ids[keep]);values.append(areas[keep])
    index=torch.from_numpy(np.stack((np.concatenate(rows),np.concatenate(cols)))) if rows else torch.empty(2,0,dtype=torch.long)
    data=torch.from_numpy(np.concatenate(values).astype('float32')) if values else torch.empty(0)
    matrix=torch.sparse_coo_tensor(index,data,(nh*nw,h*w),check_invariants=True).coalesce()
    return matrix,(nh,nw),(30.,0.,x0,0.,30.,y0),target.to_wkt()
