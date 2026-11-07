import unittest
from dataclasses import replace
import numpy as np
import torch
from .model import BandObservation,Encoder
from .adaptive_window import AdaptiveWindowEncoder,adaptive_layout,window_scale


def source(gx=.5,gy=1.21):
    return BandObservation(torch.rand(13,17),torch.ones(13,17,dtype=torch.bool),torch.tensor([450.,500.,550.]),torch.tensor([0.,1.,0.]),(gx,0,100.13,0,-gy,201.27),'EPSG:32650')


class AdaptiveChecks(unittest.TestCase):
    def test_formula_and_unique_noninteger_membership(self):
        b=source();layout=adaptive_layout(b,exponent=.5,origin_xy=(99.8,180.))
        indices=layout['indices'];self.assertEqual(len(indices),13*17);self.assertEqual(len(np.unique(indices)),len(indices))
        self.assertAlmostEqual(layout['window_pixels'][0],np.sqrt(20.))
        self.assertAlmostEqual(layout['window_pixels'][1],np.sqrt(10/1.21))
        self.assertEqual(window_scale(10.,.5),1.);self.assertEqual(window_scale(30.,.5),1.)
        for v in (0.,1.,float('nan')):
            with self.assertRaises(ValueError):window_scale(.5,v)
        coarse=source(10.,30.);c=adaptive_layout(coarse,exponent=.5,origin_xy=(99.8,-200.))
        self.assertEqual(len(c['lengths']),13*17);self.assertTrue(all(n==1 for n in c['lengths']))

    def test_support_hidden_isolation_gradient_and_checkpoint(self):
        torch.set_num_threads(2);torch.manual_seed(2);b=source();b.valid_mask[0,0]=False;b.visible=b.valid_mask.clone();b.visible[3:7,3:7]=False
        model=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64,local_batch_size=16).eval()
        r=model([b],origin_xy=(99.8,180.));changed=replace(b,reflectance=b.reflectance.clone());changed.reflectance[~b.keep()]=float('nan')
        other=model([changed],origin_xy=(99.8,180.));torch.testing.assert_close(r.tokens,other.tokens,atol=0,rtol=0)
        self.assertEqual(sum(len(c) for c in r.visible_cells),int(b.keep().sum()))
        self.assertGreater(sum(len(c) for c in r.coverage_cells),sum(len(c) for c in r.visible_cells))
        allowed=b.cells()[b.keep().flatten()]
        for c in r.receptive_cells:self.assertTrue(bool((c[:,None,:]==allowed[None,:,:]).all(-1).any(-1).all()))
        r.tokens.square().mean().backward();self.assertTrue(all(p.grad is not None for p in model.parameters()))
        self.assertFalse(any('exponent' in n for n,_ in model.named_parameters()))
        with self.assertRaises(RuntimeError):model.load_state_dict(Encoder(hyper_hidden=64).state_dict())
        invalid=replace(b,valid_mask=torch.zeros_like(b.valid_mask))
        with self.assertRaises(ValueError):model([invalid],origin_xy=(99.8,180.))

    def test_decoder_full_target_streaming(self):
        from .model import ReflectanceDecoder
        from .data import Observation
        from .tasks import spectral_task,spatial_task
        from .adaptive_streaming import prepare_adaptive_task,stream_adaptive_task
        torch.set_num_threads(2);torch.manual_seed(12)
        bands=[source(2.3,3.1) for _ in range(3)]
        for i,b in enumerate(bands):b.record_id=str(i)
        obs=Observation(bands,dict(id='adaptive_fixture',role='TRAIN'))
        for task in (spectral_task(obs,seed=0,update=0),spatial_task(obs,(105.,180.,120.,190.))):
            model=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64,local_batch_size=32).eval()
            head=ReflectanceDecoder(indexed=True,cell_indexed=True).eval()
            plan=prepare_adaptive_task(task,exponent=.5,origin_xy=(100.,160.),chunk_pixels=37,cell_indexed=True)
            r=model(task.visible,origin_xy=plan.origin_xy);losses=[]
            for t in task.targets:
                ids=torch.where((t.mask & t.band.valid_mask).flatten())[0]
                pred,valid=head(r,t.band.cells()[ids],model.srf(t.band.wavelengths_nm,t.band.response),kind=task.kind,mask_bounds=None if task.rectangle is None else r.bounds.new_tensor(task.rectangle))
                if bool(valid.any()):losses.append(((pred[valid]-t.band.reflectance.flatten()[ids][valid])/.2).square().mean())
            expected=torch.stack(losses).mean()*1.7;expected.backward()
            saved=[None if p.grad is None else p.grad.clone() for p in list(model.parameters())+list(head.parameters())]
            for chunk in (17,63):
                model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
                report=stream_adaptive_task(model,head,task,plan,train_stds={str(i):.2 for i in range(3)},chunk_pixels=chunk,loss_weight=1.7,backward=True)
                self.assertAlmostEqual(report['weighted_loss'],float(expected.detach()),places=4)
                for grad,p in zip(saved,list(model.parameters())+list(head.parameters()),strict=True):
                    if grad is None:self.assertIsNone(p.grad)
                    else:torch.testing.assert_close(p.grad,grad,atol=2e-4,rtol=2e-4)
            wrong=AdaptiveWindowEncoder(exponent=.4,hyper_hidden=64)
            with self.assertRaisesRegex(ValueError,'formula differs'):
                stream_adaptive_task(wrong,head,task,plan,backward=False)
            task.targets[0].mask.flatten()[0]=~task.targets[0].mask.flatten()[0]
            with self.assertRaisesRegex(ValueError,'changed after planning'):
                stream_adaptive_task(model,head,task,plan,backward=False)

    def test_token_count_changes_and_exact_reload(self):
        torch.set_num_threads(2);torch.manual_seed(21)
        counts=[]
        for g,side in ((.5,60),(1.21,25),(10.,3)):
            b=source(g,g);b.reflectance=torch.rand(side,side);b.valid_mask=torch.ones(side,side,dtype=torch.bool)
            layout=adaptive_layout(b,exponent=.5,origin_xy=(100.,170.))
            counts.append(len(layout['lengths']))
        self.assertGreater(counts[0],counts[1]);self.assertGreater(counts[1],counts[2])
        b=source();original=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64).eval()
        restored=AdaptiveWindowEncoder(exponent=.5,hyper_hidden=64).eval();restored.load_state_dict(original.state_dict())
        torch.testing.assert_close(original([b],origin_xy=(100.,180.)).tokens,restored([b],origin_xy=(100.,180.)).tokens,atol=0,rtol=0)
        with self.assertRaisesRegex(ValueError,'config mismatch'):
            AdaptiveWindowEncoder(exponent=.4,hyper_hidden=64).load_state_dict(original.state_dict())
        with self.assertRaises(TypeError):AdaptiveWindowEncoder(hyper_hidden=64)

    def test_training_gate_requires_merge_review(self):
        import json
        from pathlib import Path
        from .gates import require_training
        config=json.loads(Path(__file__).with_name('config.json').read_text(encoding='utf-8'))
        required={k:True for k in config['required_before_training']}
        required.pop('merge_configuration')
        with self.assertRaisesRegex(ValueError,'merge_configuration'):require_training(dict(required_before_training=required))

    def test_stopped_update_budget_remains_closed(self):
        import json,tempfile
        from pathlib import Path
        from .budget import PilotBudget
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'budget.json'
            (path.parent/'pilot_stop_decision.json').write_text(json.dumps(dict(study_sha256='a'*64,status='PRECHECK_UPDATES_STOPPED_BY_USER')))
            with self.assertRaisesRegex(RuntimeError,'stopped by user'):
                PilotBudget(path,study_sha256='a'*64).reserve('AdamW',0)
            self.assertFalse(path.exists())
