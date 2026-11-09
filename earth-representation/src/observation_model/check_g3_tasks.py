"""Final G3 real-task leakage checks and aligned preview rendering, CPU only."""
from dataclasses import replace
import shutil

import torch

from .encoder_v2 import ObservationEncoderV2,OpticalImage
from .encoder_v2_tasks import spatial_task,band_task,seed_for,input_hash
from .encoder_v2_runtime import BASE,G3,load_members,LOG_ROOT
from .g3_tasks import task_preview
from .expansion_data import ROOT,read,save,digest


def main():
    torch.set_num_threads(4);members=read(G3/'members.json');data=load_members(members)
    model=ObservationEncoderV2().eval();model.load_state_dict(torch.load(BASE/'G2_arch/initialization_untrained.pt',map_location='cpu',weights_only=True)['model'])
    tests=[];groups={};plans=read(G3/'fixed_plans.json')
    for role in ['TRAIN','VAL']:
        for sensor in ['S2','L8']:
            meta,image=next((m,im) for m,im in data if m['role']==role and m['sensor']==sensor)
            for draw in [0,1]:
                task=spatial_task(image,seed_for(meta['id'],'A',draw))
                plan=next(p for p in plans if p['member']==meta['id'] and p['kind']=='A' and p['draw']==draw)
                assert input_hash(task)==plan['input_hash']
                task.meta['encoder_grid']=plan['mask']['encoder_grid'];groups.setdefault(f'A_{role}_{draw}',[]).append((meta,image,task))
                if role=='TRAIN' and draw==0:
                    altered=[]
                    for original,visible in zip(image.bands,task.visible.bands):
                        q=original.valid.clone();indices=(~visible.visible).nonzero()
                        yy,xx=indices[0];q[yy,xx]=~q[yy,xx]
                        altered.append(replace(original,values=torch.where(visible.visible,original.values,original.values+10000),valid=q))
                    changed=spatial_task(OpticalImage(altered),seed_for(meta['id'],'A',draw))
                    assert input_hash(task)==input_hash(changed)
                    with torch.no_grad():assert torch.equal(model.encode(task.visible).features,model.encode(changed.visible).features)
                    tests.append(dict(sensor=sensor,kind='A',hidden_values_and_one_hidden_QA_exact_zero=True))
            key='B11' if sensor=='S2' else 'SR_B5';task=band_task(image,key)
            plan=next(p for p in plans if p['member']==meta['id'] and p['kind']=='B')
            with torch.no_grad():task.meta['encoder_grid']=model.encode(task.visible).grids[0]
            groups.setdefault(f'B_{role}',[]).append((meta,image,task))
            if role=='TRAIN':
                altered=OpticalImage([replace(b,values=b.values+10000,valid=b.valid.flip(0)) if b.band_key==key else b for b in image.bands])
                changed=band_task(altered,key);assert input_hash(task)==input_hash(changed)
                composite=spatial_task(image,seed_for(meta['id'],'A',0),excluded=(key,))
                assert all(b.band_key!=key for b in composite.visible.bands) and all(t.band_key!=key for t in composite.targets)
                tests.append(dict(sensor=sensor,kind='B',entire_target_removed=True,also_removed_from_composite_A=True))
    for name,entries in groups.items():
        p=G3/'preview'/(name+'.png');old=G3/'preview_before_axis_fix'/p.name;old.parent.mkdir(exist_ok=True)
        if not old.exists():shutil.copyfile(p,old)
        task_preview(entries,p,name[0]);shutil.copyfile(p,LOG_ROOT/'encoder_v2_G3_zero_update/artifacts'/p.name)
    save(G3/'leakage_tests.json',dict(status='PASS',controls=tests,fixed_validation_plans_exact=True,
        preview_axis_order_corrected=True,optimizer_updates=0))
    save(G3/'preview_source_manifest.json',dict(g3_tasks=digest(ROOT/'observation_model/g3_tasks.py'),
        check_g3_tasks=digest(ROOT/'observation_model/check_g3_tasks.py'),
        original_execution_source='source/g3_tasks.py',reason='Correct matplotlib extent order; preserve original diagrams separately'))
    print('G3_TASK_LEAKAGE_AND_PREVIEW_PASS')


if __name__=='__main__':main()
