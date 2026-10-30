"""One authorized 5000-update supplement; reuse the frozen original training loop."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import torch
import yaml
import run_use_mit_dual_routes as original
from landslide_seg.models.use_v4_reconstruction import Segmenter
from landslide_seg.models.use_v3_reconstruction import packed_state_sha256

ROOT = original.ROOT
BASE = original.BASE
CONFIG = ROOT / 'configs/use_mit_aoff.yaml'
AUDIT = BASE / 'audit/aoff'
SOURCE = BASE / 'runs/pretrain_A/u20000.pt'
RELEASE = BASE / 'releases/engineering_v2'
MARKER = BASE / 'audit/READY_FOR_TRAINING.json'
JOB = 'segment_A_off'
MODE = 'trained_A_fine_pure_RGB_context_v1'
BUILD = original.segmenter
PREFLIGHT = original.preflight


class AOffEncoder(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = original.load_encoder('A', source)
        # C0 forward skips adapter and residual, including spectral support union.
        self.source.route = 'C0'
        self.register_buffer('pure_rgb_context_mode_v1', torch.tensor(1))

    def forward(self, *args, **kwargs):
        if self.pure_rgb_context_mode_v1.item() != 1 or self.source.route != 'C0':
            raise ValueError('A-off persistent computation mode mismatch')
        return self.source(*args, **kwargs)


def segmenter(route, device='cpu', source=None):
    assert route == 'A' and Path(source) == SOURCE
    encoder = AOffEncoder(source)
    torch.manual_seed(1235)
    model = Segmenter(encoder)
    shared = torch.load(BASE/'engineering/common_initial.pt', map_location='cpu', weights_only=False)['task_head']
    current = model.state_dict()
    assert set(shared) == {k for k in current if not k.startswith('use.')}
    for name, value in shared.items():
        assert torch.equal(current[name], value), name
        current[name] = value.clone()
    model.load_state_dict(current, strict=True)
    return model.to(device)


def files():
    return [Path(__file__).resolve(), CONFIG, SOURCE, BASE/'engineering/common_initial.pt', RELEASE/'manifest.json']


def seal():
    AUDIT.mkdir(parents=True, exist_ok=True)
    assert not (BASE/'runs'/JOB/'initial.pt').exists(), 'Existing supplement; do not duplicate'
    data = dict(mode=MODE, formal_updates=5000, original_formal_updates=59000,
                cumulative_authorized_updates=64000, additional_pretrain_updates=0,
                authorization='Human authorization relayed from 01a16e71-e5a0-7113-a026-569964d1e29a: retain trained A Fine, pure RGB Context, one new 5000-step head; send to original training chat',
                files={p.relative_to(ROOT).as_posix(): original.sha(p) for p in files()})
    original.rt.write_json(AUDIT/'manifest.json', data, True)


def preflight():
    for name, digest in original.rt.read(AUDIT/'manifest.json')['files'].items():
        assert original.sha(ROOT/name) == digest, name
    config = yaml.safe_load(CONFIG.read_text(encoding='utf8'))
    assert config['aoff']['mode'] == MODE and config['aoff']['formal_updates'] == 5000
    from check_research_protocol import audit_config
    result = audit_config(CONFIG)
    original.rt.write_json(AUDIT/'protocol.json', result)
    original.verify_release(RELEASE)


def tensor_equal(a, b):
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(tensor_equal(v, b[k]) for k,v in a.items())
    if isinstance(a, (list,tuple)): return len(a) == len(b) and all(tensor_equal(x,y) for x,y in zip(a,b))
    if hasattr(a, 'shape'): return bool((a == b).all())
    return a == b


def verify(device):
    preflight(); original.pt.deterministic()
    config = yaml.safe_load(CONFIG.read_text(encoding='utf8'))
    with original.v4.guarded_device(device) as gate:
        model = segmenter('A', device, SOURCE)
        full = BUILD('A', device, SOURCE).eval()
        c0 = BUILD('C0', device).eval()
        train, _ = original.datasets(config)
        samples = [original.rt.collate([train[i]])['image'].to(device) for i in (0,17,100)]
        # RGB holes with other bands still valid exercise the support-union difference.
        masked = copy.deepcopy(samples[0])
        for i in original.rgb_indices(masked['band_ids'][0]):
            masked['pixel_valid'][0,i,24:72,24:72] = False
        samples.append(masked)
        calls = []
        hooks = [m.register_forward_hook(lambda *args: calls.append('forbidden'))
                 for m in (model.use.encoder.source.adapter, model.use.encoder.source.residual_zero)]
        checks = []
        with torch.no_grad():
            for sample in samples:
                off = model.use.encoder(sample, diagnostics=True)
                a = full.use.encoder(sample, diagnostics=True)
                c = c0.use.encoder(sample, diagnostics=True)
                assert torch.equal(off['fine'], a['fine'])
                assert torch.equal(off['valid']['fine'], a['valid']['fine'])
                for key in ('patch_tokens','mit_input'):
                    assert torch.equal(off[key],c[key]), key
                for x,y in zip(off['context'],c['context']): assert torch.equal(x,y)
                for x,y in zip(off['valid']['context'],c['valid']['context']): assert torch.equal(x,y)
                assert off['residual'] is None
                checks.append('Fine=A; tokens/Context/support=C0; residual=None')
        for hook in hooks: hook.remove()
        assert not calls
        initial = torch.load(BASE/'runs/segment_A/initial.pt',map_location='cpu',weights_only=False)
        head = {k:v for k,v in model.state_dict().items() if not k.startswith('use.')}
        assert all(torch.equal(v.cpu(),initial['model'][k]) for k,v in head.items())
        digest = hashlib.sha256()
        for name,value in sorted(head.items()):
            digest.update(str((name,str(value.dtype),tuple(value.shape))).encode())
            digest.update(value.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
        head_sha = digest.hexdigest()
        del initial,full,c0,samples
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=6e-4,weight_decay=.01)
        assert len(opt.state)==0
        reference = packed_state_sha256(original.v4.check_frozen_boundary(model,opt,'segmentation'))
        job = dict(steps=5000,warmup=250)
        records=[]
        for step in (1,2):
            model.train();original.v4.schedule(opt,step,job)
            records.append(original.v4.segment_update(model,opt,train,step,config,device))
            original.v4.verify_frozen(model,opt,'segmentation',reference)
            if step==1:
                original.v4.save_checked(model,opt,1,None,dict(mode=MODE),records.copy(),AUDIT/'recovery.pt',reference,'segmentation')
        expected = copy.deepcopy(original.pt.checkpoint(model,opt,2,None,dict(mode=MODE),records))
        original.pt.restore(model,opt,AUDIT/'recovery.pt',dict(mode=MODE))
        model.train();original.v4.schedule(opt,2,job)
        replay=original.v4.segment_update(model,opt,train,2,config,device)
        actual=original.pt.checkpoint(model,opt,2,None,dict(mode=MODE),records)
        assert records[-1]==replay
        for key in ('model','optimizer','rng'): assert tensor_equal(expected[key],actual[key]), key
        original.v4.verify_frozen(model,opt,'segmentation',reference)
        result=dict(status='PASS',mode=MODE,device_gate=gate,checks=checks,forbidden_module_calls=len(calls),
                    head_initial_sha256=head_sha,head_initial_tensors=len(head),encoder_sha256=reference,
                    optimizer_initial_states=0,engineering_updates=2,engineering_replay_updates=1,
                    formal_updates=0,exact_restore=['model','optimizer','all RNG','step2 record'],
                    manifest_sha256=original.sha(AUDIT/'manifest.json'))
        original.rt.write_json(AUDIT/'acceptance.json',result,True)
        print(json.dumps(result),flush=True)


def install():
    original.CONFIG = CONFIG
    original.matrix = lambda: {JOB:dict(route='A',mode='segmentation',steps=5000,warmup=250)}
    original.segmenter = segmenter
    original.preflight = preflight


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['seal','verify','run','evaluate'])
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path);args=parser.parse_args()
    if args.action=='seal':
        PREFLIGHT();seal();return
    if args.action=='verify':verify(args.device);return
    proof=original.rt.read(AUDIT/'acceptance.json')
    assert proof['status']=='PASS' and proof['mode']==MODE and proof['manifest_sha256']==original.sha(AUDIT/'manifest.json')
    preflight();install()
    if args.action=='run':original.execute(JOB,args.device,RELEASE,MARKER,resume=args.resume)
    else:original.evaluate(JOB,args.checkpoint,args.device)


if __name__=='__main__':main()
