"""Run only the four corrected probes, with immutable new configurations."""
import concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.stdout.reconfigure(encoding='utf-8')
OUT=ROOT/'reports/v6_execution'
OUT.mkdir(parents=True,exist_ok=True)
CFG=ROOT/'configs/segformer/v6_repairs'
CFG.mkdir(parents=True,exist_ok=True)

def run_fold(f,device):
    cfg=yaml.safe_load((ROOT/f'configs/segformer/event_cv_v5_probe/F{f}_dofa_10.yaml').read_text(encoding='utf-8'))
    s=cfg['segformer']; s['experiment']=f'DOFA-CorrectWavelengths-FrozenProbe-EventFold{f}'
    s['data']['version']='event_cv_v6_wave_repair'
    s['model']['probe']['wavelength_version']='named_bands_v6'
    cp=CFG/f'dofa_correct_wavelengths_fold{f}.yaml'
    text=yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False)
    if cp.exists() and cp.read_text(encoding='utf-8')!=text: raise RuntimeError('Config collision')
    cp.write_text(text,encoding='utf-8')
    d=ROOT/'outputs/segformer'/s['data']['version']/s['experiment']/'1234'
    command=[sys.executable,'scripts/train_segformer.py','--config',str(cp),'--device',device]
    env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    def call(args,label):
        with (OUT/f'probe_fold{f}_{label}.log').open('w',encoding='utf-8') as log:
            subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (d/'train_summary.json').exists():
        if (d/'last.pt').exists(): command+=['--resume']
        call(command,'train')
    summary=json.loads((d/'train_summary.json').read_text(encoding='utf-8'))
    assert summary['complete'] and summary['global_step']==3600
    for ck in ['best','last']:
        for split in ['val','test']:
            tag=f'{ck}_{split}'
            if (d/'eval'/f'eval_{tag}.json').exists(): continue
            args=[sys.executable,'scripts/evaluate_segformer.py','--checkpoint',str(d/f'{ck}.pt'),
                  '--split',split,'--tag',tag,'--device',device,'--save-predictions']
            if split=='test': args+=['--manifest',f'manifests/event_cv_v5/F{f}/test.jsonl']
            call(args,tag)
    return {'fold':f,'run_dir':str(d),'complete':True,'config_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}

def worker(folds,device):
    out=[]
    for f in folds:
        print(f'[start] corrected DOFA fold {f} on {device}',flush=True)
        out.append(run_fold(f,device))
        print(f'[done] corrected DOFA fold {f}',flush=True)
    return out

if __name__=='__main__':
    from landslide_seg.models.spectral_contract import wavelengths_for
    from landslide_seg.data.segformer_dataset import S2_10_BANDS
    from landslide_seg.models.frozen_probe import RGB_CH,RGB_WAVELENGTHS
    w=wavelengths_for(S2_10_BANDS)
    assert tuple(w[i] for i in RGB_CH)==RGB_WAVELENGTHS
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        futures=[ex.submit(worker,[1,3],'cuda:0'),ex.submit(worker,[2,4],'cuda:1')]
        results=[x for fut in futures for x in fut.result()]
    (OUT/'probe_repairs_complete.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('[complete] four corrected probes and best/last inner/outer evaluations',flush=True)
