"""Executable engineering evidence; never emits B's training permission."""
import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import torch
import pretrain_use_v4 as pt
import run_use_v2 as rt
from audit_use_independent_data import sha
from landslide_seg.models.use_v4_mit import V3,MIT,FRONTEND,load_visual
from landslide_seg.models.use_v3_reconstruction import packed_state_sha256


def verify(device):
    from run_use_v4 import ROOT,BASE,preflight,EXPECTED
    preflight(); began=time.time()
    result=subprocess.run([sys.executable,'-m','pytest','tests/test_use_v4_mit.py','tests/test_use_v4_masks.py',
        'tests/test_use_v4_loading.py','-q','--basetemp=outputs/use_v4/engineering/verify_tmp',
        '--junitxml=outputs/use_v4/engineering/tests.xml'],cwd=ROOT,capture_output=True,text=True)
    (BASE/'engineering/tests.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
    if result.returncode: raise ValueError('Engineering tests failed; see tests.log')
    with rt.gpu_guard(device) as gate:
        model=pt.initialize(device); splits,stats,audit,members=pt.inputs('multisensor')
        # Fixed real TRAIN observations, distinct from validation and task labels.
        rows=[splits['TRAIN'][s][0] for s in ('s2','landsat')]
        reference=packed_state_sha256(model.use.mit); evidence=[]
        for row in rows:
            original=pt.load_rows([row],stats,device)
            gradients={}
            for target in ('spectral','fine','context','gradient','consistency'):
                model.zero_grad(set_to_none=True)
                raw,counts=pt.views(model,original,123456,32)
                raw[target].backward()
                path={name:float(sum(p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None).sqrt())
                    for name,module in [('spectral',model.use.readout),('adapter',model.use.adapter),('tokenizer',model.use.tokenizer)]}
                # Fine loss also receives the fused Context path in this reconstruction task.
                if not all(v>0 and torch.isfinite(torch.tensor(v)) for v in path.values()): raise ValueError(f'Unreachable target {target}: {path}')
                if any(p.grad is not None for p in model.use.mit.parameters()): raise ValueError('MiT parameter gradient')
                gradients[target]=dict(norms=path,value=float(raw[target].detach()),counts=counts)
            diagnoses={}
            model.eval()
            with torch.no_grad():
                for mode in ('known','partial','unknown'):
                    view,hidden,summary=pt.task_view(original,'fine',654321,metadata_mode=mode)
                    feature=model.features(view); pred=model.predict(feature,view['band_metadata'],view['known_mask'],summary)
                    removed=dict(feature); removed['main']=torch.zeros_like(feature['main'])
                    zero=model.predict(removed,view['band_metadata'],view['known_mask'],summary)
                    replaced=dict(feature); replaced['main']=feature['main'].mean((-1,-2),keepdim=True).expand_as(feature['main'])
                    mean=model.predict(replaced,view['band_metadata'],view['known_mask'],summary)
                    diagnoses[mode]=dict(finite=bool(torch.isfinite(pred).all()),raw=float(pt.mean_error(pred,original['x'],hidden)),
                        zero_context_delta=float((pred-zero).abs().mean()),mean_context_delta=float((pred-mean).abs().mean()),
                        limitation='read-only response sensitivity, not a trained causal ablation')
            model.train(); evidence.append(dict(file=row['file'],sensor=row['sensor'],gradients=gradients,metadata_diagnostics=diagnoses))
        if packed_state_sha256(model.use.mit)!=reference: raise ValueError('MiT bytes changed')
        source=torch.load(V3,map_location='cpu',weights_only=False)['use']; visual=load_visual().segformer.encoder.state_dict()
        reused=[]
        for name,value in model.use.state_dict().items():
            if name.split('.')[0] in FRONTEND:
                if not torch.equal(value.cpu(),source[name]): raise ValueError('Frontend mismatch '+name)
                reused.append(dict(name=name,origin='v3',shape=list(value.shape),frozen=False))
            elif name.startswith('mit.'):
                if not torch.equal(value.cpu(),visual[name[4:]]): raise ValueError('MiT mismatch '+name)
                reused.append(dict(name=name,origin='MiT safetensors',shape=list(value.shape),frozen=True,
                    active=not name.startswith('mit.patch_embeddings.0.proj.')))
            elif not name.startswith('adapter.'): raise ValueError('Unclassified encoder tensor '+name)
        rt.write_json(BASE/'engineering/loading.json',dict(tensors=reused,random=['adapter','reconstruction fusion/decode/condition'],
            skipped=['MiT RGB patch_embeddings.0.proj (loaded and frozen, deliberately bypassed)','ADE20K 150-class classifier','v3 inject/merges/stages/projections/feedback_gate/local'],source_sha256={str(p.relative_to(ROOT)):d for p,d in EXPECTED.items()}))
        rt.write_json(BASE/'engineering/verification.json',dict(status='PASS',device_gate=gate,tests=ET.parse(BASE/'engineering/tests.xml').getroot().attrib,
            real_train_checks=evidence,mit_sha256=reference,members=members,audit=audit,seconds=time.time()-began,
            discarded_optimizer_updates_per_pytest_run=10,formal_updates=0,engineering_trial_updates='separate trial200',
            mask_semantics='support is any original visible pixel in conv/SR receptive field; mixed receptive fields retain zero-fill semantics'))
    print('PASS: tests, source reuse, real TRAIN targets and frozen gradients')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:0'); a=p.parse_args(); torch.set_num_threads(4); verify(a.device)
