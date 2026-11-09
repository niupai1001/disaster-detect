"""Synthetic LR contracts: no real updates or data mutations."""
import copy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from encoder_v3.model import BandObservation
from encoder_v3.data import Observation
from encoder_v3.latent_response import SensorKey,ChannelCatalog,ChannelObservation
from encoder_v3.latent_response.geometry import FullGeometry,position_features
from encoder_v3.latent_response.model import LatentResponseModel
from encoder_v3.latent_response.fusion import LocalEmbed,ChannelFusion
from encoder_v3.latent_response.data import make_view,pool_recipe
from encoder_v3.latent_response.execution import view_loss,hierarchical_loss
from encoder_v3.latent_response.decoder import exact_knn
from encoder_v3.latent_response.checkpoint import parameter_sha,save_checkpoint,load_checkpoint,validation_transaction,rng_state
from encoder_v3.latent_response.train import require_training

KEY=SensorKey('S2','S2A')
CAT=ChannelCatalog.from_ordered(['B02','B03','B04'])


def fixture(side=24):
    torch.manual_seed(9)
    model=LatentResponseModel({KEY:CAT})
    channels=[]; bands=[]
    for name in CAT.channel_ids:
        values=torch.rand(side,side); valid=torch.ones_like(values,dtype=torch.bool)
        affine=(10.,0.,600000.,0.,-10.,4500000.)
        channels.append(ChannelObservation(KEY,CAT.meta(name),values.clone(),valid.clone(),valid.clone(),affine,'EPSG:32631'))
        bands.append(BandObservation(values.clone(),valid.clone(),torch.tensor([400.,500.]),torch.ones(2),affine,'EPSG:32631',name))
    return model,channels,Observation(bands,dict(id='synthetic',sensor='S2',instrument='S2A',role='TRAIN'))


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_local_embed_hard_mask_and_order(self):
        embed=LocalEmbed(); u=torch.randn(2,16,32); support=torch.ones(2,16); support[:,0]=0
        original=embed(u,support); u[:,0]=float('nan')
        torch.testing.assert_close(embed(u,support),original)
        self.assertFalse(torch.allclose(embed(u.flip(1),support.flip(1)),original))

    def test_dynamic_R_and_empty_context(self):
        fusion=ChannelFusion()
        for c,r in ((1,1),(2,2),(4,2),(10,4),(224,15),(257,16)):
            _,s,l,mask,_=fusion(torch.randn(1,c,128),torch.ones(1,c,16),torch.randn(c,32))
            self.assertEqual(int(mask.sum()),r)
        with self.assertRaisesRegex(ValueError,'NO_VISIBLE_CONTEXT'):
            fusion(torch.zeros(1,2,128),torch.zeros(1,2,16),torch.zeros(2,32))

    def test_hidden_full_patch_and_gradient_path(self):
        model,channels,obs=fixture(); view=make_view(obs,channels,'A')
        representation=model.encode(view.channels,view.geometry,tile_rows=8)
        self.assertLess(len(representation.fine),9)
        seen=[]
        handle=model.fusion.register_forward_hook(lambda mod,args,out:(out[2].retain_grad(),seen.append(out[2])) and None)
        result=view_loss(model,view,backward=True,chunk_pixels=37,tile_rows=8)
        handle.remove()
        self.assertGreater(result['targets'],0)
        self.assertTrue(torch.isfinite(seen[0].grad).all())
        self.assertGreater(float(seen[0].grad.abs().sum()),0)
        for module in (model.registry,model.memory,model.module_query,model.kernels,model.embed,model.fusion,model.spatial,model.decoder):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()))

    def test_A_B1_poison_output_and_gradient_equivalence(self):
        for kind in ('A','B1'):
            model,channels,obs=fixture(16); view=make_view(obs,channels,kind)
            poisoned=copy.deepcopy(view)
            for c in poisoned.channels:c.values[~c.keep()]=float('nan')
            first=view_loss(model,view,backward=True,chunk_pixels=64,tile_rows=8)
            gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            second=view_loss(model,poisoned,backward=True,chunk_pixels=64,tile_rows=8)
            self.assertAlmostEqual(first['loss'],second['loss'],places=6)
            for n,p in model.named_parameters():
                if n in gradients:torch.testing.assert_close(p.grad,gradients[n],atol=1e-5,rtol=1e-4)

    def test_B1_geometry_and_target_value_isolation(self):
        model,channels,obs=fixture(); view=make_view(obs,channels,'B1')
        self.assertNotIn('B04',[c.channel_meta.channel_id for c in view.channels])
        self.assertEqual(view.geometry,FullGeometry.from_channels(channels))
        r=model.encode(view.channels,view.geometry)
        channels[-1].values.fill_(9999)
        second=make_view(obs,channels,'B1'); r2=model.encode(second.channels,second.geometry)
        torch.testing.assert_close(r.fine,r2.fine)

    def test_tile_chunk_order_and_batch_equivalence(self):
        model,channels,_=fixture(72); geometry=FullGeometry.from_channels(channels)
        first=model.encode(channels,geometry,tile_rows=72,chunk_rows=None)
        second=model.encode(list(reversed(channels)),geometry,tile_rows=13,chunk_rows=1)
        for field in ('fine','latent','global_latent'):
            torch.testing.assert_close(getattr(first,field),getattr(second,field),atol=1e-5,rtol=1e-4)
        pts=first.centers[:5]
        y,_=model.predict(first,CAT.meta('B02'),(10.,10.),pts,chunk_pixels=5)
        a=torch.autograd.grad(y.sum(),tuple(model.parameters()),allow_unused=True)
        z,_=model.predict(second,CAT.meta('B02'),(10.,10.),pts,chunk_pixels=2)
        b=torch.autograd.grad(z.sum(),tuple(model.parameters()),allow_unused=True)
        torch.testing.assert_close(y,z,atol=1e-5,rtol=1e-4)
        for x,y in zip(a,b):
            if x is not None:torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-3)
        batch=model.encode_batch([(channels,geometry),(channels,geometry)],tile_rows=24)
        torch.testing.assert_close(batch[0].fine,batch[1].fine)

    def test_mixed_native_gsd_and_offset_grid_streaming(self):
        model,cs,_=fixture(32)
        old=cs[1];cs[1]=ChannelObservation(KEY,old.channel_meta,old.values[::2,::2].clone(),
            old.valid[::2,::2].clone(),old.visible[::2,::2].clone(),
            (20.,0.,600005.,0.,-20.,4500005.),old.crs)
        old=cs[2];cs[2]=ChannelObservation(KEY,old.channel_meta,old.values[:,::2].clone(),
            old.valid[:,::2].clone(),old.visible[:,::2].clone(),
            (20.,0.,600000.,0.,-10.,4500000.),old.crs)
        geometry=FullGeometry.from_channels(cs)
        self.assertEqual(geometry.patch_size,80.)
        a=model.encode(cs,geometry,tile_rows=32,chunk_rows=None)
        b=model.encode(cs,geometry,tile_rows=7,chunk_rows=1)
        torch.testing.assert_close(a.fine,b.fine,atol=1e-5,rtol=1e-4)
        self.assertFalse(a.census['full_scene_attention'])

    def test_target_geometry_dependency(self):
        model,channels,_=fixture(); g=FullGeometry.from_channels(channels); r=model.encode(channels,g)
        pts=r.centers[:1].clone().requires_grad_()
        prediction,_=model.predict(r,CAT.meta('B02'),(10.,10.),pts)
        grad=torch.autograd.grad(prediction.sum(),pts)[0]
        self.assertGreater(float(grad.abs().sum()),0)
        smaller,_=model.predict(r,CAT.meta('B02'),(20.,10.),pts.detach())
        self.assertFalse(torch.allclose(prediction,smaller))

    def test_position_reference_and_crs_rejection(self):
        model,cs,_=fixture(); g=FullGeometry.from_channels(cs); _,pts=g.patch_centers('cpu')
        self.assertFalse(torch.allclose(position_features(pts,g),position_features(pts+2048,g)))
        changed=copy.deepcopy(cs); changed[0].crs='EPSG:32632'
        with self.assertRaises(ValueError):FullGeometry.from_channels(changed)
        changed=copy.deepcopy(cs); changed[0].affine=(10.,0.,0.,0.,-10.,100.)
        with self.assertRaises(ValueError):FullGeometry.from_channels(changed)

    def test_exact_knn_counts_and_ties(self):
        centers=torch.tensor([[-1.,0.],[1.,0.],[0.,3.]])
        ids,metrics=exact_knn(torch.tensor([[0.,0.],[0.,2.]]),centers,k=2)
        self.assertEqual(ids.tolist(),[[0,1],[2,0]])
        self.assertEqual(metrics['distance_evaluations'],6)

    def test_streamed_loss_equals_direct_gradient(self):
        model,cs,obs=fixture(16); view=make_view(obs,cs,'B1')
        streamed=view_loss(model,view,backward=True,chunk_pixels=17,tile_rows=8)
        grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        rep=model.encode(view.channels,view.geometry,tile_rows=8)
        from encoder_v3.latent_response.data import target_chunks
        loss=0
        for c,mask,truth in view.targets:
            pts,y=next(target_chunks(c,mask,truth,10000))
            predicted,_=model.predict(rep,c.channel_meta,c.gsd_xy,pts,chunk_pixels=256)
            loss=loss+(predicted-y).square().mean()/len(view.targets)
        loss.backward()
        self.assertAlmostEqual(streamed['loss'],float(loss.detach()),places=5)
        for n,p in model.named_parameters():
            if n in grads:torch.testing.assert_close(p.grad,grads[n],atol=1e-4,rtol=1e-3)

    def test_hierarchical_reduction_and_no_pixel_cap(self):
        self.assertEqual(hierarchical_loss([('S2',2.,4.),('S2',4.,6.),('Landsat',6.,8.)]),5.5)
        recipe=pool_recipe([dict(id='large',role='TRAIN',sensor='S2',bands=[{},{}],shape=[9999,9999])])
        self.assertEqual(recipe['member_count'],1)
        with self.assertRaises(PermissionError):require_training({'real_updates_authorized':True})

    def test_checkpoint_validation_resume_and_identity_rejection(self):
        model,cs,obs=fixture(16); model.double()
        for c in cs:c.values=c.values.double()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        state=dict(cursor=0,validation={})
        identity=dict(code='code',config='config',registry=model.registry.identity_sha256(),catalog='catalog',data='data',sampling='sampling')
        def update():
            optimizer.zero_grad(set_to_none=True)
            view_loss(model,make_view(obs,cs,'B1'),backward=True,chunk_pixels=256)
            optimizer.step(); state['cursor']+=1
        update()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            original=parameter_sha(model)
            with self.assertRaises(RuntimeError):
                with validation_transaction(model,optimizer,state):
                    model.eval(); state['cursor']=999
                    with torch.no_grad():next(model.parameters()).add_(1)
                    torch.rand(3); raise RuntimeError('validation failure')
            self.assertEqual(parameter_sha(model),original)
            self.assertEqual(state['cursor'],1)
            state['validation']={'completed':True,'metric':1.2}
            save_checkpoint(path,model,optimizer,state,identity)
            random_expected=torch.rand(3)
            update(); expected=parameter_sha(model); expected_parameters=copy.deepcopy(model.state_dict())
            load_checkpoint(path,model,optimizer,state,identity)
            torch.testing.assert_close(torch.rand(3),random_expected,rtol=0,atol=0)
            self.assertTrue(state['validation']['completed'])
            self.assertEqual(state['cursor'],1)
            update()
            maximum=max(float((model.state_dict()[n]-v).abs().max()) for n,v in expected_parameters.items())
            print('resume_next_update_max_abs',maximum)
            for n,v in expected_parameters.items():torch.testing.assert_close(model.state_dict()[n],v,atol=1e-10,rtol=1e-8)
            expected=parameter_sha(model)
            for field in identity:
                bad=dict(identity,**{field:'changed'})
                with self.assertRaises(ValueError):load_checkpoint(path,model,optimizer,state,bad)
                self.assertEqual(parameter_sha(model),expected)
