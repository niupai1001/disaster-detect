"""Shared local I/O and TVLP plumbing for the gated V2 tasks."""
from datetime import datetime,timezone
import json
from pathlib import Path
import random

import numpy as np
import torch

from src.ssl.tvlp_writer import TvlpRun
from .expanded_dataset import G1OpticalDataset
from .encoder_v2 import image_from_g1
from .expansion_data import ROOT,OUT,read

BASE=ROOT/'observation_model/outputs/encoder_v2'
G3=BASE/'G3_task';G4=BASE/'G4_fit';LOG_ROOT=BASE/'tvlp'
VIEWER=Path('C:/Users/yzh/Downloads/training_viewer_v1.1_spec_and_examples/training_viewer_v1/.venv/Scripts/training-viewer.exe')
VIEWER_DATA=ROOT/'observation_model/outputs/repair_v2/viewer'


def selected_members(per_sensor=2):
    result=[]
    for role in ['TRAIN','VAL']:
        dataset=G1OpticalDataset(OUT,BASE/'G1',role)
        for sensor,band_count in [('S2',10),('L8',7)]:
            candidates=sorted([r for r in dataset.rows if r['sensor']==sensor and r['band_count']==band_count],
                key=lambda r:(sum(np.prod(b['shape']) for b in r['bands']),r['id']))
            groups=set();chosen=[]
            for row in candidates:
                if row['release_isolation_group'] in groups:continue
                groups.add(row['release_isolation_group']);chosen.append(row)
                if len(chosen)==per_sensor:break
            if len(chosen)!=per_sensor:raise ValueError('Not enough isolated controls')
            result+=chosen
    return result


def load_members(members,device='cpu'):
    datasets={role:G1OpticalDataset(OUT,BASE/'G1',role) for role in ['TRAIN','VAL']}
    indices={role:{r['id']:i for i,r in enumerate(ds.rows)} for role,ds in datasets.items()}
    return [(row,image_from_g1(datasets[row['role']][indices[row['role']][row['id']]],device)) for row in members]


def definitions():
    result={}
    for phase in ['diagnostic_A','diagnostic_B','train_A','train_B','val_A','val_B','fixed_train_A','fixed_train_B','train_new_mask_A',
        'unseen_response_A','unseen_response_B','combo_SWIR_pair','combo_rededge_group','combo_visible_NIR_pair','combo_visible_NIR_group']:
        result[phase+'/score']={'direction':'min'}
        for sensor,bands in [('S2',['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']),('L8',['SR_B'+str(i) for i in range(1,8)])]:
            for band in bands:
                for metric in ['nmse','rmse','mae','bias','prediction_variance','target_variance']:
                    result[f'{phase}/{sensor}/{band}/{metric}']={'direction':'min' if metric in ['nmse','rmse','mae'] else 'none'}
    for phase,keys in dict(state=['optimizer_updates','learning_rate','lambda_A','lambda_B','remaining_updates','resume_count','gradient_norm'],
        capacity=['source_observations','encoder_views','target_pixels','visible_pixels','native_pixels','tokens','local_feature_vectors','encoder_feature_elements','peak_gpu_MiB','forward_backward_seconds','source_observations_per_second'],
        gradients=['A_norm','B_norm'],baseline=['A_mean','A_spatial','B_mean','B_ridge']).items():
        for key in keys:result[phase+'/'+key]={'direction':'none'}
    return result


class V2Run(TvlpRun):
    def __init__(self,run_id,name,config,resume=False,sequence=0):
        self.sequence=sequence
        super().__init__(LOG_ROOT,project='统一遥感EncoderV2',name=name,run_id=run_id,
            experiment='G5原生局部特征与冻结迁移' if 'G5' in run_id else 'G3/G4有界任务',seed=0,metric_definitions=definitions(),config=config,resume=resume)

    def log(self,phase,metrics,global_step=0,wall_time_sec=None):
        self.sequence+=1
        row=dict(session_id=self.session_id,event_id=f'{self.run_id}:metric:{self.sequence}',
            timestamp=datetime.now(timezone.utc).isoformat(),phase=phase,global_step=int(global_step),
            metrics={k:float(v) for k,v in metrics.items()},extensions=dict(writer_sequence=self.sequence))
        if wall_time_sec is not None:row['wall_time_sec']=float(wall_time_sec)
        self._metrics.write(json.dumps(row,ensure_ascii=False)+'\n');self._metrics.flush()

    def artifact(self,path,kind='image',tag='task-preview',phase='diagnostic',step=0):
        if not (self.dir/path).is_file():raise FileNotFoundError(path)
        event=dict(type='artifact',timestamp=datetime.now(timezone.utc).isoformat(),session_id=self.session_id,
            path=path,artifact_type=kind,tag=tag,phase=phase,global_step=step,role='comparison')
        self._events.write(json.dumps(event,ensure_ascii=False)+'\n');self._events.flush()


def checkpoint(path,model,optimizer,state,run,device='cpu'):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None,
        viewer_sequence=run.sequence,run_id=run.run_id,data_manifest_sha256=read(OUT/'release.json')['manifest_sha256'])
    temp=path.with_suffix('.tmp');torch.save(payload,temp);temp.replace(path)


def restore(path,model,optimizer,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['data_manifest_sha256']==read(OUT/'release.json')['manifest_sha256']
    model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['torch_rng']);np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
    if payload['cuda_rng'] is not None:torch.cuda.set_rng_state(payload['cuda_rng'],device)
    return payload
