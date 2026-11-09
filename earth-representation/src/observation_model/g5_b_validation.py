"""Read-only fixed B-combination panel, separate from frozen G5 selection."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time

import torch

from .encoder_v2 import EncoderConfig,ObservationEncoderV2
from .encoder_v2_tasks import spectral_task,run_task,aggregate_metrics
from .g5_protocol import G5,ObservationStream,statistics
from .check_encoder_v2 import state_hash
from .expansion_data import ROOT,read,save,digest


def prepare():
    destination=G5/'B_combination_validation';destination.mkdir(exist_ok=True)
    contract=destination/'contract.json'
    if contract.exists():return destination,read(contract)
    stream=ObservationStream();panels=[]
    definitions=[
        ('S2',['B02','B03'],'train_distribution',4),
        ('S2',['B06','B07'],'train_distribution',10),
        ('S2',['B08','B8A'],'train_distribution',10),
        ('S2',['B02','B03','B04'],'train_distribution',10),
        ('L8',['SR_B1','SR_B2'],'train_distribution',7),
        ('L8',['SR_B5','SR_B6'],'train_distribution',7),
        ('L8',['SR_B2','SR_B3','SR_B4'],'train_distribution',7),
        ('S2',['B11','B12'],'unseen_combination',10),
        ('S2',['B05','B06','B07'],'unseen_combination',10),
        ('S2',['B04','B08'],'unseen_combination',4),
        ('L8',['SR_B6','SR_B7'],'unseen_combination',7),
        ('L8',['SR_B3','SR_B4','SR_B5'],'unseen_combination',7)]
    for sensor,hidden,role,minimum_bands in definitions:
        members=[stream.val.rows[i]['id'] for i in stream.primary if stream.val.rows[i]['sensor']==sensor
            and stream.val.rows[i]['band_count']>=minimum_bands
            and set(hidden)<{b['band_key'] for b in stream.val.rows[i]['bands']}]
        assert members
        panels.append(dict(id=sensor+'__'+'_'.join(hidden),sensor=sensor,hidden=hidden,role=role,members=members))
    exposure=Counter()
    for line in (G5/'exposure.jsonl').open(encoding='utf-8'):
        row=json.loads(line)
        if row['task']=='B':exposure[('S2' if row['instrument'].startswith('S2') else 'L8')+':'+','.join(sorted(row['target_bands']))]+=1
    for panel in panels:
        panel['observed_training_exposures_at_declaration']=exposure[panel['sensor']+':'+','.join(sorted(panel['hidden']))]
        assert (panel['observed_training_exposures_at_declaration']>0)==(panel['role']=='train_distribution')
    result=dict(panels=panels,validation_pixels_per_target=1024,model_updates=0,
        declaration_after_training_started=True,declared_at_training_update=read(G5/'status.json')['optimizer_updates'],
        source_sha256=digest(Path(__file__)),training_sampler_unchanged=True,selection_unchanged=True,
        single_band_validation='Existing primary/B reports each sensor and target independently every500 updates',
        scope='Twelve explicitly listed grouped combinations, not every arbitrary randomized training subset',
        evaluation_schedule='Latest available500-step checkpoint, then checkpoints at2000-step multiples and final best/last; no training interruption')
    save(contract,result);return destination,result


@torch.no_grad()
def evaluate(filename='last.pt',check=False):
    destination,contract=prepare();config=read(G5/'config.json')
    for relative,sha in read(G5/'source_manifest.json')['sha256'].items():assert digest(ROOT/relative)==sha
    torch.set_num_threads(4);torch.cuda.set_device(config['device']);stream=ObservationStream(config['device'])
    path=G5/filename;payload=torch.load(path,map_location='cpu',weights_only=False);step=payload['state']['update']
    output=destination/(f'check_step_{step:05d}.json' if check else f'{Path(filename).stem}_step_{step:05d}.json')
    if output.exists():print('ALREADY_EVALUATED',output);return
    model=ObservationEncoderV2(EncoderConfig(local_width=16,query_chunk=2048)).to(config['device']).eval()
    model.load_state_dict(payload['model']);before=state_hash(model);started=time.perf_counter();results=[]
    indices={r['id']:i for i,r in enumerate(stream.val.rows)}
    for panel in contract['panels']:
        rows=[];resources=[];by_visible=defaultdict(list)
        members=panel['members']
        if check:members=[min(members,key=lambda m:sum(b['shape'][0]*b['shape'][1] for b in stream.val.rows[indices[m]]['bands']))]
        for member in members:
            meta,image=stream.image(indices[member],'VAL');task=spectral_task(image,panel['hidden'])
            assert set(panel['hidden']).isdisjoint(b.band_key for b in task.visible.bands)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                r,resource,_=run_task(model,task,meta,statistics(),target_sample_pixels=contract['validation_pixels_per_target'],logical_update=0)
            visible_keys=','.join(sorted(b.band_key for b in task.visible.bands))
            rows+=r;by_visible[visible_keys]+=r
            resources.append(dict(member=member,input_hash=resource['input_hash'],target_pixels=resource['target_pixels'],visible_bands=visible_keys))
        results.append(dict(combination=panel['id'],hidden=panel['hidden'],role=panel['role'],observations=len(members),
            metrics=aggregate_metrics(rows),by_visible_band_set={k:aggregate_metrics(v) for k,v in by_visible.items()},members=resources))
        print('B_FIXED_COMBINATION',step,panel['id'],len(members),flush=True)
    assert before==state_hash(model)
    save(output,dict(status='PASS' if check else 'COMPLETE',checkpoint_update=step,checkpoint=filename,rows=results,
        model_hash_before=before,model_hash_after=state_hash(model),encoder_optimizer_updates=0,
        contract_sha256=digest(destination/'contract.json'),source_sha256=digest(Path(__file__)),seconds=time.perf_counter()-started,
        selection_changed=False,scope='Engineering subset only' if check else 'All eligible fixed primary VAL members per combination'))
    if not check:
        from src.ssl.tvlp_writer import TvlpRun
        from .encoder_v2_runtime import LOG_ROOT
        run_id=config['run_id']+'__B_fixed_panel'
        definitions={row['combination']+'/'+key:dict(direction='min' if key.endswith(('nmse','rmse','mae','score')) else 'none')
            for row in results for key in row['metrics']}
        run=TvlpRun(LOG_ROOT,project='统一遥感EncoderV2',name='G5固定组合验证｜训练内与未见组合',
            run_id=run_id,resume=(LOG_ROOT/run_id/'run.json').exists(),metric_definitions=definitions,
            config=dict(contract_sha256=digest(destination/'contract.json'),optimizer_updates=0,selection_changed=False))
        for row in results:run.log(row['combination'],row['metrics'],global_step=step)
        import csv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        assets=run.dir/'artifacts';assets.mkdir(exist_ok=True)
        fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
        labels=[r['combination'].replace('S2__','S2 ').replace('L8__','L8 ').replace('SR_','') for r in results]
        values=[r['metrics']['score'] for r in results]
        ax.barh(labels,values,color=['#3c75b5' if r['role']=='train_distribution' else '#d79035' for r in results]);ax.invert_yaxis()
        ax.set_xlabel('Fixed combination NMSE (lower is better)');ax.set_title(f'Checkpoint {step}: blue=train-distribution, orange=unseen combinations')
        for i,value in enumerate(values):ax.text(value,i,f' {value:.4f}',va='center')
        filename=f'B_combinations_{step:05d}.png';fig.savefig(assets/filename,dpi=150);plt.close(fig)
        table=f'B_combinations_{step:05d}.csv'
        with (assets/table).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['combination','role','observations','NMSE'])
            for row in results:writer.writerow([row['combination'],row['role'],row['observations'],row['metrics']['score']])
        from datetime import datetime,timezone
        for name,kind in [(filename,'image'),(table,'table')]:
            run._events.write(json.dumps(dict(type='artifact',timestamp=datetime.now(timezone.utc).isoformat(),session_id=run.session_id,
                path='artifacts/'+name,artifact_type=kind,tag='固定12组合真实结果',phase='validation',global_step=step,role='comparison'),ensure_ascii=False)+'\n')
        run._events.flush()
        run.finish('completed','Read-only combination validation; no optimizer or checkpoint selection changes')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',default='last.pt');parser.add_argument('--check',action='store_true');parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:print(prepare()[0])
    else:evaluate(args.checkpoint,args.check)
