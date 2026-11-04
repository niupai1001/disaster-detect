"""Frozen G5 membership, sampling and task contract; no optimizer entry point."""
from collections import defaultdict,Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .encoder_v2 import EncoderConfig,image_from_g1
from .encoder_v2_tasks import training_spatial_task,training_spectral_keys,spectral_task
from .encoder_v2_runtime import BASE
from .expanded_dataset import G1OpticalDataset
from .audit_ridge_bands import normalization
from .expansion_data import ROOT,OUT,read,save,digest

G5=BASE/'G5_native_bank'


def curriculum_g5(update):
    t=min(max(update/2000,0),1);smooth=t*t*(3-2*t)
    return 1-.75*smooth,1.


@lru_cache(maxsize=1)
def statistics():
    norm=normalization();result={}
    for instrument,fam in [('S2A','S2'),('S2B','S2'),('S2C','S2'),('LC08','L8')]:
        for key,(mean,std) in norm.items():
            f,band=key.split(':')
            if f==fam:result[instrument+':'+band]=dict(mean=mean,std=std,source='S2A/B or LC08 TRAIN only; no S2C normalization fit')
    return result


class ObservationStream:
    def __init__(self,device='cpu'):
        self.device=device;self.train=G1OpticalDataset(OUT,BASE/'G1','TRAIN');self.val=G1OpticalDataset(OUT,BASE/'G1','VAL')
        self.indices=[i for i,r in enumerate(self.train.rows) if r['instrument'] in ['S2A','S2B','LC08']]
        self.primary=[i for i,r in enumerate(self.val.rows) if r['instrument'] in ['S2A','S2B','LC08']]
        self.response=[i for i,r in enumerate(self.val.rows) if r['instrument']=='S2C']
        self.eligible=defaultdict(list)
        for index in self.indices:
            row=self.train.rows[index]
            for band in row['bands']:self.eligible[(row['sensor'],band['band_key'])].append(index)
        self.band_keys={sensor:sorted({b for s,b in self.eligible if s==sensor}) for sensor in ['S2','L8']}
        self.rng=np.random.default_rng(0)

    def draw(self,update):
        result=[];selected=set()
        for sensor in ['S2','L8']:
            names=self.band_keys[sensor]
            for slot in range(16):
                anchor=names[((update-1)*16+slot)%len(names)]
                candidates=self.eligible[sensor,anchor]
                for _ in range(100):
                    index=int(self.rng.choice(candidates))
                    if index not in selected:break
                if index in selected:raise ValueError('Could not build unique source batch')
                selected.add(index);result.append((index,anchor))
        return result

    def image(self,index,role='TRAIN'):
        dataset=self.train if role=='TRAIN' else self.val
        return dataset.rows[index],image_from_g1(dataset[index],self.device)

    def tasks(self,index,anchor,update):
        row,image=self.image(index);hidden=training_spectral_keys(image,row['id'],update,anchor)
        return row,image,[training_spatial_task(image,row['id'],update,excluded=hidden),spectral_task(image,hidden)]


def prepare():
    G5.mkdir(exist_ok=True)
    if (G5/'config.json').exists():return read(G5/'config.json')
    stream=ObservationStream();known=[stream.train.rows[i] for i in stream.indices]
    primary=[stream.val.rows[i] for i in stream.primary];response=[stream.val.rows[i] for i in stream.response]
    assert len(known)==6402 and len(primary)==367 and len(response)==16
    assert not ({r['release_isolation_group'] for r in known}&{r['release_isolation_group'] for r in primary+response})
    for name,rows in [('train_members.jsonl',known),('validation_members.jsonl',primary),('response_members.jsonl',response)]:
        (G5/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    save(G5/'statistics.json',statistics())
    config=dict(run_id='encoder_v2_G5_nativebank_20k',seed=0,maximum_updates=20000,
        authorization='User: 20000步预算、可以早停; start only after pre_G5 engineering/throughput gates',
        effective_source_batch=32,microbatch=1,source_sensor_quota=dict(S2=16,L8=16),
        observation_sampling='Uniform legal members conditional on round-robin real anchor band; distinct observations within each batch',
        target_sampling='Fresh training spatial masks and spectral subsets per member/logical update; fixed validation masks',
        train_count=6402,primary_validation_count=367,response_validation_count=16,
        trained_instruments=['S2A','S2B','LC08'],unseen_response='S2C; never used for fitting, LR, early stop, checkpoint selection',
        reserved='Original469 S2C TRAIN and524 LC09 TRAIN remain unchanged and excluded from this transfer experiment; no data deleted',
        optimizer='AdamW',lr=3e-4,weight_decay=.04,clip=3.,warmup_updates=500,minimum_lr=1e-6,
        architecture=dict(local_width=16,query_chunk=2048,coarse_width=128,spatial_blocks=4,max_geo_tokens=8192),
        precision=read(BASE/'pre_G5/precision_and_timing.json')['chosen_precision'],
        geometry_and_loss_precision='FP32',train_target_samples_per_band=2048,validation_target_samples_per_band=1024,
        target_scope='All real valid pixels of full native bands; uniform without-replacement sampling is an unbiased loss estimator, not an image crop',
        task_weights='lambdaB1 from first step; lambdaA smooth1->0.25 during first2000 steps, then0.25',
        curriculum_change_reason='The G3 schedule was a candidate. G4 provides no support for7000 A-only G5 steps; pixel-conditioned B starts alongside A.',
        supervision='TRAIN-only normalized reflectance MSE; mean_sensor mean_band mean_observation mean_valid_pixel',
        validation_interval=500,challenge_interval=2000,checkpoint_interval=500,
        selection='Absolute lowest fixed primary (0.25*A + B)/1.25; fixed throughout training',
        scheduler=dict(significant_relative_improvement=.001,plateau_evaluations=4,factor=.5),
        early_stop=dict(minimum_updates=2000,stale_evaluations=12,lr_at_most_initial_divided_by=16),
        convergence_rule='Budget stop while improving is unconfirmed convergence; no termination merely for failing Ridge in first1000 updates',
        training_combo_holdouts=['S2 B11+B12','S2 B05+B06+B07','S2 B04+B08','L8 SR_B6+SR_B7','L8 SR_B3+SR_B4+SR_B5'],
        feature_probe='After unlabeled best selection: TRAIN-only full-target linear probes on frozen H/H+pooled native-bank vs matched random initializer; measure error reduction beyond Ridge on known, unseen combinations and S2C',
        probe_not_selection=True,physical_losses=False,semantic_labels=False,other_training_groups=0,
        device='cuda:1',peak_memory_cap_bytes=24*1024**3,network_image_download=False,
        old_execution_policy_sha256=digest(ROOT/'observation_model/execution_policy.json'),
        source_data_manifest_sha256=read(OUT/'release.json')['manifest_sha256'],
        train_manifest_sha256=digest(G5/'train_members.jsonl'),validation_manifest_sha256=digest(G5/'validation_members.jsonl'),
        statistics_sha256=digest(G5/'statistics.json'),automatic_extension_after_budget=False)
    save(G5/'config.json',config)
    save(G5/'approval_state.json',dict(stage='G5',user_authorized_max_updates=20000,technical_gate='PENDING_FULL_BATCH_AND_MAX_SOURCE_CHECK',
        started=False,old_queue_revival=False,source_data_role_changes=0))
    save(G5/'membership_summary.json',dict(train=dict(Counter(r['instrument'] for r in known)),
        primary_val=dict(Counter(r['instrument'] for r in primary)),response_val=dict(Counter(r['instrument'] for r in response))))
    # Frozen probes use physical target meanings selected before G5 outcomes.
    plans=[]
    for sensor,target,context in [('S2','B08','RGB_only'),('S2','B11','single'),('S2','B07','single'),('L8','SR_B5','single')]:
        for role,rows in [('TRAIN',known),('VAL',primary)]:
            eligible=[r for r in rows if r['sensor']==sensor and target in [b['band_key'] for b in r['bands']] and (context=='RGB_only' or r['band_count']==(10 if sensor=='S2' else 7))]
            eligible.sort(key=lambda r:hashlib.sha256(('probe0:'+r['id']).encode()).hexdigest())
            selected=[];groups=set()
            for row in eligible:
                if row['release_isolation_group'] in groups:continue
                groups.add(row['release_isolation_group']);selected.append(row['id'])
                if len(selected)==16:break
            plans.append(dict(sensor=sensor,target=target,context=context,role=role,members=selected,pixels_per_member=128))
    save(G5/'feature_probe_plan.json',dict(plans=plans,regularization=.01,seed=0,
        task='Fit full normalized targets with TRAIN-only linear heads on frozen features; Ridge is comparison, not an input. H128/H+pooled-bank192 keep fixed dimensions for unseen combinations. Compare trained/random encoders; never choose checkpoint from this score.',
        unseen_S2C='Apply RGB-only B08 probe trained on S2A/B to original16 S2C VAL; no S2C head fitting'))
    return config


if __name__=='__main__':print(prepare())
