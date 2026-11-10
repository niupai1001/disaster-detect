"""Real native-data scale comparison, with no optimizer updates or implicit L."""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from .data import load_data, build_model, sample_from_dataset, ROOT
from .geometry import metric_geometry
from .train import write


def extent(row):
    boxes=[]
    for band in row['bands']:
        a,_,x,_,e,y=band['transform'];h,w=band['shape']
        boxes.append((x,y+e*h,x+a*w,y))
    return max(b[2] for b in boxes)-min(b[0] for b in boxes),max(b[3] for b in boxes)-min(b[1] for b in boxes)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--census-only',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--only-id')
    parser.add_argument('--scales',type=float,nargs='+',default=(32.,64.,128.,256.))
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh evidence directory required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(17)
    datasets,resolver=load_data();scales=args.scales
    census=[];selected=[]
    for role,ds in datasets.items():
        for i,row in enumerate(ds.rows):
            w,h=extent(row)
            census.append(dict(id=row['id'],role=role,sensor=row['sensor'],width_m=w,height_m=h,
                channels=len(row['bands']),tokens={str(int(L)):math.ceil(w/L)*math.ceil(h/L) for L in scales}))
    write(args.output/'census.json',census)
    for sensor in ('S2','L8','EnMAP'):
        rows=[(i,r) for i,r in enumerate(datasets['TRAIN'].rows) if r['sensor']==sensor]
        rows.sort(key=lambda item:math.prod(extent(item[1])))
        for label,index in (('small',0),('median',len(rows)//2),('largest',len(rows)-1)):
            i,row=rows[index];selected.append((label,i,row))
    report=dict(status='RUNNING',optimizer_updates=0,scope='Engineering scale diagnostic; no selected production L',
                cases=[],selected=[dict(label=k,id=r['id'],sensor=r['sensor']) for k,_,r in selected])
    write(args.output/'report.json',report)
    if args.census_only:return
    base=json.loads((ROOT/'configs/config_shared_mae_v1.json').read_text())
    for label,index,row in selected:
        if args.only_id and row['id']!=args.only_id:continue
        sample=sample_from_dataset(datasets['TRAIN'],index,resolver,args.device)
        for L in scales:
            config=dict(base,patch_metres=L)
            model=build_model(config).to(args.device)
            case=dict(id=row['id'],sensor=row['sensor'],size=label,L=L,tasks={})
            try:
                geometry=metric_geometry(sample.channels,L);case['tokens']=math.prod(geometry.shape)
                # Bound the actual ordered frontend, not a hypothetical unfused attention matrix.
                free,_=torch.cuda.mem_get_info()
                lower_bound=case['tokens']**2*4*4
                frontend=case['tokens']*model.layout**2*33*4*len(sample.channels)
                if frontend>free:
                    case.update(status='RESOURCE_LIMIT',reason='Ordered FP32 frontend tensor alone exceeds current free VRAM',
                        attention_score_bytes=lower_bound,ordered_frontend_bytes=frontend,free_bytes=free)
                    report['cases'].append(case);write(args.output/'report.json',report)
                    print(row['sensor'],label,L,case['status'],flush=True)
                    del model;torch.cuda.empty_cache();continue
                start=time.perf_counter();floors=[]
                with torch.no_grad():
                    for channel in sample.channels:
                        if not bool(channel.valid.any()):continue
                        mapping=model.area_cache.get(channel.grid,geometry,model.layout,args.device)
                        field,_=mapping.aggregate(channel.values.reshape(1,-1,1),channel.valid.reshape(1,-1))
                        restored=mapping.native_prediction(field.reshape(-1,model.layout**2))
                        mask=channel.valid.flatten()
                        floors.append(float((restored[mask]-channel.values.flatten()[mask]).square().mean()))
                case['layout_roundtrip_mse']=sum(floors)/len(floors)
                case['layout_seconds']=time.perf_counter()-start
                for kind in model.available_tasks(sample):
                    model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    loss,details=model([sample],kind=kind,seed=17,epoch=0)
                    loss.mean().backward();torch.cuda.synchronize()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.,error_if_nonfinite=True)
                    case['tasks'][kind]=dict(loss=float(loss.detach()),seconds=time.perf_counter()-start,
                        peak_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(norm),
                        census={k:v for k,v in details[0].items() if k not in ('predictions','target_masks')})
                case['status']='PASS'
            except (ValueError,RuntimeError) as exc:
                case.update(status='FAILED',reason=str(exc))
            report['cases'].append(case);write(args.output/'report.json',report)
            print(row['sensor'],label,L,case['status'],case.get('reason',''),flush=True)
            del model;torch.cuda.empty_cache()
    report['status']='COMPLETE';write(args.output/'report.json',report)


if __name__=='__main__':main()
