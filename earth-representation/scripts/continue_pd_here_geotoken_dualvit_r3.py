"""Explicitly authorized 700-parent, ten-epoch continuation of R3 paired G2."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
from train_pd_here_geotoken_dualvit_r3 import ROOT,model_for,update,evaluate
from pd_here_geotoken_dualvit_r3.data import OpticalDataset,geometry,load_window,batch,masks,write_json
from encoder_v3.latent_response.tvlp_writer import RunWriter
from pd_here_geotoken_dualvit_r3.evaluation import metric_records, reduce_records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    source=ROOT/'outputs/pd_here_geotoken_dualvit_r3';out=source/'subset700_10epochs'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists():raise ValueError('Run exists; do not overwrite completed updates')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    panel=json.loads((source/'window_plan.json').read_text());dataset=OpticalDataset(ROOT)
    rows={role:[json.loads(l) for l in (source/f'{role}.jsonl').read_text().splitlines()] for role in ('train','val')}
    layout,_=geometry(rows['train'][0]);cached={}
    for role in rows:
        for row in rows[role]:cached[row['id']]=load_window(dataset,row,panel[row['id']]['origin'],layout)
        print(f'Loaded {role}: {len(rows[role])}',flush=True)
    model,opt=model_for(layout,args.device)
    state=torch.load(source/'paired_last.pt',map_location=args.device,weights_only=False)
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    identity=dict(parent_run=str(source),start_step=200,epochs=10,train_parents=700,development_val_parents=300,
                  batch=16,steps_per_epoch=44,updates=440,seed=17,lr=1e-4,precision='FP32',
                  initialization='continue paired_last.pt including AdamW state',
                  windows='fixed registered initial native windows; no window rotation in this bounded comparison',
                  schedule='constant existing lr; no reset/warmup',
                  authorization='user explicitly selected 700 TRAIN, 10 epochs, 440 updates',
                  selection='last; no TEST or downstream claim')
    write_json(out/'identity.json',identity)
    def inputs(selected,exposure):
        ids=[r['id'] for r in selected];win=[panel[i]['window_id'] for i in ids]
        values,qa=batch([cached[i] for i in ids],args.device)
        return values,qa,masks(qa,ids,win,exposure)
    @torch.no_grad()
    def score(role):
        records = {'A': [], 'B1': []}
        model.eval()
        for start in range(0,len(rows[role]),16):
            selected=rows[role][start:start+16];values,qa,views=inputs(selected,0)
            for task,view in zip(records,views,strict=True):
                prediction=model(values,qa,*view)['prediction']
                for record in metric_records(prediction,values,qa,*view,task):
                    record['parent'] += start
                    records[task].append(record)
        model.train()
        result = {}
        for task,items in records.items():
            values,counts = reduce_records(items,counts=True)
            result.update({f'{task}_{key}':value for key,value in values.items()})
            result.update({f'{task}_{key}_parents':value for key,value in counts.items()})
        return result
    definitions={key:dict(label=key,direction='min',role='loss',unit='') for key in
                 ('train_a','train_b','val_a','val_b','val_a_centered','val_b_centered','val_a_gradient','val_b_gradient')}
    definitions['epoch_seconds']=dict(label='完整epoch秒数（含VAL和保存）',direction='none',role='other',unit='s')
    viewer=source/'viewer/subset700_10epochs'
    history=[]
    try:
        with RunWriter(viewer,experiment=dict(id='pd_here_geotoken_dualvit_r3',name='GeoToken DualViT R3'),
                       variant=dict(id='subset700_10epochs',name='700 TRAIN · paired续训10epoch',kind='baseline'),
                       metrics=definitions,charts=['train_a','train_b','val_a','val_b'],numbers=['epoch_seconds'],config=identity) as log:
            before=dict(train=score('train'),val=score('val'));write_json(out/'before.json',before)
            log.log(0,dict(val_a=before['val']['A_mse'],val_b=before['val']['B1_mse']),epoch=0)
            for epoch in range(1,11):
                torch.cuda.synchronize(args.device);tick=time.perf_counter();order=list(rows['train'])
                random.Random(17+epoch).shuffle(order);total={'A':0.,'B1':0.};step=0
                for start in range(0,len(order),16):
                    selected=order[start:start+16];values,qa,views=inputs(selected,200+epoch-1)
                    losses,_=update(model,opt,values,qa,views,'paired');step+=1
                    for task,value in losses.items():total[task]+=value*len(selected)
                    write_json(out/'status.json',dict(status='TRAINING',epoch=epoch,step=step,steps_per_epoch=44,total_updates=(epoch-1)*44+step))
                torch.cuda.synchronize(args.device);train_seconds=time.perf_counter()-tick
                val=score('val')
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=200+epoch*44,
                                identity=identity,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'last.pt')
                torch.cuda.synchronize(args.device);seconds=time.perf_counter()-tick
                record=dict(epoch=epoch,train_seconds=train_seconds,epoch_seconds=seconds,
                            train={t:v/700 for t,v in total.items()},val=val)
                history.append(record);write_json(out/'history.json',history)
                log.log(epoch*44,dict(train_a=total['A']/700,train_b=total['B1']/700,val_a=val['A_mse'],val_b=val['B1_mse'],
                                     val_a_centered=val['A_centered_mse'],val_b_centered=val['B1_centered_mse'],
                                     val_a_gradient=val['A_gradient_mse'],val_b_gradient=val['B1_gradient_mse'],epoch_seconds=seconds),epoch=epoch)
                print(json.dumps(record),flush=True)
            after=dict(train=score('train'),val=history[-1]['val']);write_json(out/'after.json',after)
            write_json(out/'status.json',dict(status='COMPLETED',epochs=10,updates=440,checkpoint=str(out/'last.pt'),
                                             mean_epoch_seconds=sum(r['epoch_seconds'] for r in history)/10))
    except Exception as error:
        write_json(out/'status.json',dict(status='FAILED',error=repr(error)));raise

if __name__=='__main__':main()
