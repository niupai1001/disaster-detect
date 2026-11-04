"""Batch independent observation views through shared attention; same state dict."""
from collections import defaultdict,Counter
import time

import torch
from torch.nn import functional as F

from .encoder_v2 import token_grid,pool_native,position_features,EncoderRepresentation,NativeGridDecoder
from .encoder_v2_fast_geometry import GeometryFastObservationEncoderV2
from .encoder_v2_tasks import run_task,sample_target_indices,training_spatial_task,spectral_task
from .g5_protocol import G5,statistics,curriculum_g5
from .g5_train import planned_batch
from .g4_fit import append


class BatchedObservationEncoderV2(GeometryFastObservationEncoderV2):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.batch_geometry_cache={}
        def nearest(bank,coordinates,bounds):
            key=(bank.neighbor_geometry_key,id(coordinates),tuple(bounds))
            if key not in self.batch_geometry_cache:
                self.batch_geometry_cache[key]=(coordinates,NativeGridDecoder.nearest_native(self.decoder,bank,coordinates,bounds))
            return self.batch_geometry_cache[key][1]
        self.decoder.nearest_native=nearest
        encode=self.encode
        def encode_single(images,excluded_bands=()):
            self.batch_geometry_cache.clear()
            return encode(images,excluded_bands)
        self.encode=encode_single

    def encode_views(self,images):
        self.batch_geometry_cache.clear()
        encoder=self.encoder;prepared=[]
        for image in images:
            bands=[b for b in image.bands if bool(b.keep().any())]
            assert bands and len({b.crs for b in bands})==1
            bounds=(min(b.bounds[0] for b in bands),min(b.bounds[1] for b in bands),max(b.bounds[2] for b in bands),max(b.bounds[3] for b in bands))
            grid=token_grid(bounds,min(min(b.support) for b in bands),self.config);grid['crs']=bands[0].crs
            features=[];keeps=[];support=[]
            for band in bands:
                moments,valid,coverage=pool_native(band,grid,self.config)
                features.append(encoder.projection(moments,band.srf,band.support)*valid[:,None]);keeps.append(valid)
                support.append(dict(band_key=band.band_key,native_pixel_support_m=band.support,native_shape=tuple(band.values.shape),native_transform=band.transform,source_coverage_fraction=coverage.detach()))
            keys=torch.stack(features,1);mask=torch.stack(keeps,1);keep=mask.any(1)
            ny,nx=grid['shape'];left,bottom,right,top=bounds
            yy,xx=torch.meshgrid(torch.arange(ny,device=keys.device),torch.arange(nx,device=keys.device),indexing='ij')
            xy=torch.stack([(xx.flatten()+.5)/nx,(yy.flatten()+.5)/ny],-1).to(keys.dtype)
            q=(encoder.query+encoder.position(position_features(xy)))[:,None]
            ex=torch.linspace(left,right,nx+1,device=keys.device,dtype=torch.float64);ey=torch.linspace(top,bottom,ny+1,device=keys.device,dtype=torch.float64)
            footprint=torch.stack([ex[xx.flatten()],ey[yy.flatten()+1],ex[xx.flatten()+1],ey[yy.flatten()]],-1)
            banks=[encoder.native_bank(b,grid) for b in bands]
            prepared.append(dict(q=q,keys=keys,mask=mask,keep=keep,grid=grid,footprint=footprint,support=support,banks=banks))
        maximum=max(p['keys'].shape[1] for p in prepared)
        q=torch.cat([p['q'] for p in prepared]);keys=torch.cat([F.pad(p['keys'],(0,0,0,maximum-p['keys'].shape[1])) for p in prepared])
        mask=torch.cat([F.pad(p['mask'],(0,maximum-p['mask'].shape[1])) for p in prepared])
        fused=encoder.fusion(q,keys,mask)[:,0].split([len(p['q']) for p in prepared])
        xs=[x*p['keep'][:,None] for x,p in zip(fused,prepared)]
        for index,block in enumerate(encoder.blocks):
            windows=[];masks=[];geometry=[]
            for x,p in zip(xs,prepared):
                ny,nx=p['grid']['shape'];d=self.config.width;window=min(self.config.window,ny,nx);shift=window//2 if index%2 else 0
                bottom=(-(ny+shift))%window;right=(-(nx+shift))%window
                z=F.pad(x.reshape(ny,nx,d),(0,0,shift,right,shift,bottom));valid=F.pad(p['keep'].reshape(ny,nx),(shift,right,shift,bottom));hh,ww=valid.shape
                z=z.reshape(hh//window,window,ww//window,window,d).permute(0,2,1,3,4).reshape(-1,window*window,d)
                valid=valid.reshape(hh//window,window,ww//window,window).permute(0,2,1,3).reshape(-1,window*window)
                windows.append(z);masks.append(valid);geometry.append((ny,nx,hh,ww,window,shift,len(z)))
            # Different small native windows may have different attention lengths.
            maximum=max(z.shape[1] for z in windows)
            result=block(torch.cat([F.pad(z,(0,0,0,maximum-z.shape[1])) for z in windows]),torch.cat([F.pad(v,(0,maximum-v.shape[1])) for v in masks]))
            xs=[]
            for z,(ny,nx,hh,ww,window,shift,n) in zip(result.split([g[-1] for g in geometry]),geometry):
                z=z[:,:window*window].reshape(hh//window,ww//window,window,window,self.config.width).permute(0,2,1,3,4).reshape(hh,ww,self.config.width)
                xs.append(z[shift:shift+ny,shift:shift+nx].reshape(ny*nx,self.config.width))
        output=[]
        for x,p in zip(xs,prepared):
            f=p['footprint'];scale=((f[:,2]-f[:,0])*(f[:,3]-f[:,1])).clamp_min(0).sqrt()
            output.append(EncoderRepresentation(x[None],f[None],p['keep'][None],scale[None],[p['support']],[p['grid']],optional_local_features=[p['banks']]))
        return output


def query_chunks(model,representation,targets,indices):
    """Prepare each view independently; mixing is only along attention batch axis."""
    decoder=model.decoder;grid=representation.grids[0];ny,nx=grid['shape'];left,bottom,right,top=grid['bounds']
    tokens=representation.features[0,:ny*nx];valid=representation.valid[0,:ny*nx];context=tokens[valid].mean(0)
    coordinates=[];conditions=[];supports=[];lengths=[]
    for target,ids in zip(targets,indices):
        a,b,c,d,e,f=target.transform;assert not b and not d and target.crs==grid['crs'];h,w=target.shape
        coordinates.append(torch.stack([(ids%w).double()*a+(c-left)+a/2,(ids//w).double()*(-e)+(top-f)-e/2],-1));lengths.append(len(ids))
        condition=decoder.condition(torch.cat([target.srf.to(tokens.device),torch.log(tokens.new_tensor([a,-e])/10)]))
        conditions.append(condition.expand(len(ids),-1));supports.append(tokens.new_tensor([a,-e]).expand(len(ids),-1))
    coordinates=torch.cat(coordinates);conditions=torch.cat(conditions);supports=torch.cat(supports);xy=(coordinates/coordinates.new_tensor([right-left,top-bottom])).to(tokens.dtype)
    center_x=torch.floor(xy[:,0]*nx).long();center_y=torch.floor(xy[:,1]*ny).long();offsets=torch.tensor([(dy,dx) for dy in [-1,0,1] for dx in [-1,0,1]],device=tokens.device)
    chunks=[]
    for begin in range(0,len(xy),8192):
        end=min(begin+8192,len(xy));yy=center_y[begin:end,None]+offsets[:,0];xx=center_x[begin:end,None]+offsets[:,1]
        inside=(yy>=0)&(yy<ny)&(xx>=0)&(xx<nx);keys=yy.clamp(0,ny-1)*nx+xx.clamp(0,nx-1);keep=inside&valid[keys];k=tokens[keys]*keep[...,None];empty=~keep.any(1)
        k=k.clone();k[:,0]=torch.where(empty[:,None],context,k[:,0]);keep=keep.clone();keep[:,0]|=empty
        coordinates_chunk=coordinates[begin:end].float();scale=supports[begin:end].float();local=[]
        for bank in representation.optional_local_features[0]:
            count=min(decoder.config.local_neighbors,len(bank.features))
            with torch.no_grad():best=decoder.nearest_native(bank,coordinates_chunk,grid['bounds'])
            relative=bank.relative_xy_m[best]-coordinates_chunk[:,None]
            relative=torch.cat([torch.tanh(relative/(4*scale[:,None])),torch.log(coordinates_chunk.new_tensor(bank.native_pixel_support_m)/scale)[:,None].expand(-1,count,-1)],-1)
            local.append(decoder.local_key(bank.features[best])+decoder.local_relative(relative))
        local=torch.cat(local,1);k=torch.cat([k,local],1);keep=torch.cat([keep,torch.ones(local.shape[:2],device=tokens.device,dtype=torch.bool)],1)
        q=(conditions[begin:end]+decoder.position(position_features(xy[begin:end])))[:,None];observed=keep.any(1)&(xy[begin:end]>=0).all(1)&(xy[begin:end]<=1).all(1)
        chunks.append((q,k,keep,observed))
    return chunks,lengths


def batched_tasks(model,jobs,stats,samples,update):
    representations=model.encode_views([task.visible for _,task in jobs]);prepared=[];lengths=[]
    for representation,(row,task) in zip(representations,jobs):
        indices=[sample_target_indices(t,row['id'],task.kind,samples,update) for t in task.targets]
        chunks,lens=query_chunks(model,representation,[t.grid for t in task.targets],indices);prepared.append(chunks);lengths.append(lens)
    predictions=[[] for _ in jobs]
    # Group one query chunk from each observation; no attention crosses examples.
    for offset in range(max(map(len,prepared))):
        selected=[(i,chunks[offset]) for i,chunks in enumerate(prepared) if offset<len(chunks)]
        maximum=max(chunk[1].shape[1] for _,chunk in selected)
        q=torch.cat([chunk[0] for _,chunk in selected]);k=torch.cat([F.pad(chunk[1],(0,0,0,maximum-chunk[1].shape[1])) for _,chunk in selected]);keep=torch.cat([F.pad(chunk[2],(0,maximum-chunk[2].shape[1])) for _,chunk in selected])
        for block in model.decoder.blocks:q=block(q,k,keep)
        output=model.decoder.output(q[:,0])[:,0]
        for (i,chunk),value in zip(selected,output.split([len(chunk[0]) for _,chunk in selected])):predictions[i].append((value*chunk[3],chunk[3]))
    results=[];encode=model.encode;forward=model.decoder.forward
    try:
        for representation,(row,task),parts,lens in zip(representations,jobs,predictions,lengths):
            outputs=list(zip(torch.cat([v for v,_ in parts]).split(lens),torch.cat([v for _,v in parts]).split(lens)));cursor=iter(outputs)
            model.encode=lambda image:representation
            model.decoder.forward=lambda *args,**kwargs:next(cursor)
            results.append(run_task(model,task,row,stats,target_sample_pixels=samples,logical_update=update))
    finally:model.encode=encode;model.decoder.forward=forward
    return results


def batch_backward(model,stream,update,config,log=None,benchmark=False,microbatch=2):
    plans,weights=planned_batch(stream,update);rows_by_kind=defaultdict(list);a,b=curriculum_g5(update);model.train();model.zero_grad(set_to_none=True);exposure=Counter()
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();stats=statistics()
    for begin in range(0,len(plans),microbatch):
        group=plans[begin:begin+microbatch];sources=[stream.image(p['index']) for p in group]
        for kind in ['A','B']:
            if log:log.heartbeat(update-1)
            jobs=[(row,training_spatial_task(image,row['id'],update,excluded=p['hidden']) if kind=='A' else spectral_task(image,p['hidden'])) for p,(row,image) in zip(group,sources)]
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                results=batched_tasks(model,jobs,stats,config['train_target_samples_per_band'],update)
                loss=sum(r['loss']*weights[kind,r['sensor'],r['band']] for rows,_,_ in results for r in rows)*(a if kind=='A' else b)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite microbatched objective')
            loss.backward()
            for (row,task),(rows,resource,_) in zip(jobs,results):
                for r in rows:r['loss']=r['loss'].detach()
                rows_by_kind[kind]+=rows;exposure['views']+=1;exposure['target_band_observations']+=len(rows);exposure['supervised_pixels']+=resource['target_pixels'];exposure['declared_valid_target_pixels']+=resource['full_valid_target_pixels'];exposure['native_local_vectors']+=resource['local_feature_vectors'];exposure['encoder_feature_elements']+=resource['encoder_feature_elements'];exposure['geo_tokens']+=resource['tokens'];exposure['capped_geo_views']+=int(resource['token_budget_capped'])
                if not benchmark:append(G5/'exposure.jsonl',dict(update=update,member=row['id'],instrument=row['instrument'],task=kind,target_bands=[r['band'] for r in rows],mask_seed=task.meta.get('seed'),resource=resource))
            del results,loss
        exposure['source_observations']+=len(sources)
    torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    if peak>config['peak_memory_cap_bytes']:raise RuntimeError('Microbatch exceeded frozen memory cap')
    return rows_by_kind,dict(exposure),dict(seconds=seconds,peak_bytes=peak),plans
