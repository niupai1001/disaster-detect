"""Explicit zero-update engineering cohort, never a formal training sampler."""
import math
from .data import to_device
from .tasks import spectral_task,spatial_task
from .streaming import prepare_task
from .joint import PlannedView


def prepare_batch(dataset,*,patch_m):
    selected=[]
    for instrument,count in (('S2',4),('L8',4)):
        candidates=[i for i,row in enumerate(dataset.rows) if row['sensor']==instrument and (len(row['bands'])==4 if instrument=='S2' else len(row['bands']) in (6,7))]
        if len(candidates)<count:raise ValueError('Insufficient intact diagnostic members')
        selected.extend(candidates[:count])
    observations=[dataset[i] for i in selected];plans=[];rectangles=[];stds=[]
    for observation in observations:
        first=observation.bands[0];h,w=first.reflectance.shape;a,_,left,_,e,top=first.affine;bottom=top+e*h
        origin=(left,bottom)
        # This deterministic rectangle is a declared diagnostic recipe only.
        # Actual area is recorded; it does not freeze the training distribution.
        width=max(1,round(w*a/(2*patch_m)))*patch_m;height=math.ceil(h*(-e)/patch_m)*patch_m
        rectangle=(left,bottom,left+width,bottom+height);rectangles.append(rectangle)
        stds.append(dataset.engineering_train_stds(observation))
        tasks=(spatial_task(observation,rectangle),spectral_task(observation,seed=0,update=0))
        plans.append([prepare_task(task,patch_m=patch_m,origin_xy=origin,chunk_pixels=512) for task in tasks])
    return observations,rectangles,plans,stds


def device_views(observations,rectangles,plans,stds,device):
    views=[]
    for observation,rectangle,pair,normalization in zip(observations,rectangles,plans,stds,strict=True):
        moved=to_device(observation,device)
        tasks=(spatial_task(moved,rectangle),spectral_task(moved,seed=0,update=0))
        for task,plan in zip(tasks,pair,strict=True):views.append(PlannedView(task,plan,normalization,observation.metadata['sensor']))
    return views
