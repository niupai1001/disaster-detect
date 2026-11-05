import unittest
from dataclasses import replace
import torch
from .model import Encoder,ReflectanceDecoder,Representation
from .test_model import band


class ExecutionChecks(unittest.TestCase):
    def test_physical_metadata_and_decoder_boundary(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        sources=[band(),band()];sources[0].record_id='audit-only-0';sources[1].record_id='audit-only-1'
        sources[1].affine=(.37,0,0,0,-1.21,6);sources[0].valid_mask[1,1]=False
        model=Encoder(hyper_hidden=64,local_batch_size=8,native_mode='grouped',layout_mode='indexed').eval()
        r=model(sources,patch_m=3.1,origin_xy=(0,0));n=len(r.tokens)
        self.assertEqual(r.gsd_xy.shape,(n,2));self.assertEqual(r.srf_condition.shape,(n,32))
        self.assertEqual(len(r.source_bands),n);self.assertEqual(r.merge_scale_m,3.1);self.assertEqual(r.origin_xy,(0,0))
        self.assertEqual(len(r.full_input_cells),n)
        self.assertTrue(all(a is b for a,b in zip(r.full_input_cells,r.receptive_cells)))
        complete=torch.unique(torch.cat(r.full_input_cells),dim=0)
        visible=torch.unique(torch.cat([source.cells()[source.keep().flatten()] for source in sources]),dim=0)
        torch.testing.assert_close(complete,visible)
        torch.testing.assert_close(r.support_xy,r.bounds[:,2:]-r.bounds[:,:2])
        for source in sources:
            index=torch.tensor([name==source.record_id for name in r.source_bands])
            torch.testing.assert_close(r.gsd_xy[index],torch.tensor([source.affine[0],-source.affine[4]]).expand(int(index.sum()),-1))
            condition=model.srf(source.wavelengths_nm,source.response)
            torch.testing.assert_close(r.srf_condition[index],condition.expand(int(index.sum()),-1))
        renamed=[replace(source,record_id='renamed-'+str(i)) for i,source in enumerate(sources)]
        other=model(renamed,patch_m=3.1,origin_xy=(0,0))
        torch.testing.assert_close(other.tokens,r.tokens,atol=0,rtol=0)
        self.assertNotEqual(other.source_bands,r.source_bands)
        decoder=ReflectanceDecoder(indexed=True,cell_indexed=True);target=sources[0].cells()
        condition=model.srf(sources[0].wavelengths_nm,sources[0].response)
        a,sa=decoder(r,target,condition,kind='B1')
        changed=replace(r,gsd_xy=r.gsd_xy*10,srf_condition=r.srf_condition+1000,source_bands=('ignored',)*n,merge_scale_m=10000.)
        b,sb=decoder(changed,target,condition,kind='B1')
        self.assertTrue(torch.equal(sa,sb));torch.testing.assert_close(a[sa],b[sb],atol=0,rtol=0)
        gradients=torch.autograd.grad(r.srf_condition.sum(),tuple(model.srf.parameters()))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))

    def test_cell_index_multicell_area_holes_and_invalid_tokens(self):
        cells=[torch.tensor(v,dtype=torch.float64) for v in ([[0,0,1,1],[2,0,3,1]],[[.5,0,2.5,1]],[[0,0,3,1]])]
        z=Representation(torch.randn(3,128,requires_grad=True),cells,torch.tensor([[1.5,.5]]*3,dtype=torch.float64),torch.tensor([[0,0,3,1],[.5,0,2.5,1],[0,0,3,1]],dtype=torch.float64),torch.tensor([True,True,False]),'EPSG:32631')
        targets=torch.tensor([[0,0,3,1],[1.1,.2,1.9,.8],[3.5,0,4.5,1]],dtype=torch.float64)
        condition=torch.randn(32,requires_grad=True)
        dense=ReflectanceDecoder();candidate=ReflectanceDecoder(indexed=True,cell_indexed=True);candidate.load_state_dict(dense.state_dict())
        a,sa=dense(z,targets,condition,kind='B2');b,sb=candidate(z,targets,condition,kind='B2')
        self.assertTrue(torch.equal(sa,torch.tensor([True,True,False])));self.assertTrue(torch.equal(sa,sb))
        torch.testing.assert_close(a[sa],b[sb],atol=2e-6,rtol=2e-5)
        ga=torch.autograd.grad(a[sa].sum(),(z.tokens,condition,dense.mlp[0].weight),retain_graph=True)
        gb=torch.autograd.grad(b[sb].sum(),(z.tokens,condition,candidate.mlp[0].weight))
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(gb[0][2],torch.zeros_like(gb[0][2]),atol=0,rtol=0)

    def test_indexed_layout_support_and_gradients(self):
        torch.set_num_threads(2);a=Encoder(hyper_hidden=64,local_batch_size=8).eval()
        b=Encoder(hyper_hidden=64,local_batch_size=8,layout_mode='indexed').eval();b.load_state_dict(a.state_dict())
        source=band();source.affine=(.37,0,10,0,-1.21,20);source.valid_mask[1,2]=False
        source.visible=source.valid_mask.clone();source.visible[2:4,3:]=False
        x=a([source],patch_m=1.5,origin_xy=(10,10));y=b([source],patch_m=1.5,origin_xy=(10,10))
        torch.testing.assert_close(x.tokens,y.tokens,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(x.centers,y.centers);torch.testing.assert_close(x.bounds,y.bounds)
        for p,q in zip(x.cells,y.cells):torch.testing.assert_close(p,q)
        for p,q in zip(x.receptive_cells,y.receptive_cells):torch.testing.assert_close(p,q)
        x.tokens.square().mean().backward();y.tokens.square().mean().backward()
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)

    def test_native_tiling_and_grouped_gradients(self):
        torch.set_num_threads(2)
        reference=Encoder(hyper_hidden=64).eval()
        sources=[band(),band()];sources[0].valid_mask[2,2]=False;sources[0].reflectance[2,2]=float('nan')
        expected=[reference.native(b)[0] for b in sources]
        sum(x.square().mean() for x in expected).backward()
        for mode,rows in [('serial',2),('serial',1),('grouped',None)]:
            candidate=Encoder(hyper_hidden=64,native_mode=mode,tile_rows=rows).eval();candidate.load_state_dict(reference.state_dict())
            actual=[p[0] for p in candidate.native_grouped(sources)] if mode=='grouped' else [candidate.native(b)[0] for b in sources]
            for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
            sum(x.square().mean() for x in actual).backward()
            for a,b in zip(reference.parameters(),candidate.parameters()):
                if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=2e-4)

    def test_exact_union_hole_does_not_overlap(self):
        cells=torch.tensor([[0.,0.,1.,1.],[2.,0.,3.,1.]],dtype=torch.float64)
        z=Representation(torch.ones(1,128),[cells],torch.tensor([[1.5,.5]],dtype=torch.float64),torch.tensor([[0.,0.,3.,1.]],dtype=torch.float64),torch.ones(1,dtype=torch.bool),'EPSG:32631')
        for decoder in (ReflectanceDecoder(),ReflectanceDecoder(indexed=True,cell_indexed=True)):
            output,supported=decoder(z,torch.tensor([[1.1,.1,1.9,.9]],dtype=torch.float64),torch.ones(32),kind='B2')
            self.assertFalse(bool(supported.any()));self.assertTrue(bool(torch.isnan(output).all()))

    def test_cross_sample_batch_isolation(self):
        model=Encoder(hyper_hidden=64,local_batch_size=8).eval()
        x=torch.randn(2,5,128);y=x.clone();y[1]+=100
        with torch.no_grad():
            a=model.local(x)[0];b=model.local(y)[0]
        torch.testing.assert_close(a,b)

    def test_observation_batch_geometry_gradients_and_isolation(self):
        torch.set_num_threads(2);torch.manual_seed(31)
        model=Encoder(hyper_hidden=64,local_batch_size=8,native_mode='grouped',layout_mode='indexed').eval()
        first=band();second=band();second.affine=(.37,0,10,0,-1.21,20)
        second.valid_mask[1,2]=False;second.visible=second.valid_mask.clone();second.visible[2:4,3:]=False
        second.reflectance=second.reflectance.detach().requires_grad_(True)
        items=[([first],20,(0,0)),([second],3.1,(10,10))]
        expected=[model(b,patch_m=p,origin_xy=o) for b,p,o in items]
        actual=model.forward_batch(items)
        self.assertNotEqual(len(actual[0].tokens),len(actual[1].tokens))
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a.tokens,b.tokens,atol=2e-6,rtol=2e-5)
            torch.testing.assert_close(a.centers,b.centers);torch.testing.assert_close(a.bounds,b.bounds)
            for x,y in zip(a.cells,b.cells):torch.testing.assert_close(x,y)
            for x,y in zip(a.receptive_cells,b.receptive_cells):torch.testing.assert_close(x,y)
            torch.testing.assert_close(a.gsd_xy,b.gsd_xy);torch.testing.assert_close(a.srf_condition,b.srf_condition)
            torch.testing.assert_close(a.support_xy,b.support_xy);self.assertEqual(a.source_bands,b.source_bands)
        parameters=tuple(model.parameters())
        reference=torch.autograd.grad(sum(r.tokens.square().mean() for r in expected),parameters)
        candidate=torch.autograd.grad(sum(r.tokens.square().mean() for r in actual),parameters,retain_graph=True)
        for a,b in zip(reference,candidate):torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-4)
        leakage=torch.autograd.grad(actual[0].tokens.sum(),second.reflectance)[0]
        torch.testing.assert_close(leakage,torch.zeros_like(leakage),atol=0,rtol=0)
        second.reflectance=second.reflectance.detach()+100
        changed=model.forward_batch(items)
        torch.testing.assert_close(changed[0].tokens,actual[0].tokens,atol=0,rtol=0)

    def test_sparse_decoder_outputs_and_gradients(self):
        encoder=Encoder(hyper_hidden=64,local_batch_size=8).eval();source=band()
        z=encoder([source],patch_m=20,origin_xy=(0,0))
        dense=ReflectanceDecoder();sparse=ReflectanceDecoder(indexed=True);sparse.load_state_dict(dense.state_dict())
        cells=ReflectanceDecoder(indexed=True,cell_indexed=True);cells.load_state_dict(dense.state_dict())
        condition=encoder.srf(source.wavelengths_nm,source.response)
        targets=torch.cat((source.cells(),source.cells()+1000))
        for kind in ('A','B1','B2'):
            mask=targets.new_tensor([20,20,40,40]) if kind=='A' else None
            dense.zero_grad();sparse.zero_grad()
            a,va=dense(z,targets,condition,kind=kind,mask_bounds=mask)
            b,vb=sparse(z,targets,condition,kind=kind,mask_bounds=mask)
            self.assertTrue(torch.equal(va,vb));torch.testing.assert_close(a[va],b[vb],atol=2e-6,rtol=2e-5)
            ga=torch.autograd.grad(a[va].sum(),(z.tokens,condition,dense.mlp[0].weight),retain_graph=True)
            gb=torch.autograd.grad(b[vb].sum(),(z.tokens,condition,sparse.mlp[0].weight),retain_graph=True)
            for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)
            c,vc=cells(z,targets,condition,kind=kind,mask_bounds=mask)
            self.assertTrue(torch.equal(va,vc));torch.testing.assert_close(a[va],c[vc],atol=2e-6,rtol=2e-5)
            gc=torch.autograd.grad(c[vc].sum(),(z.tokens,condition,cells.mlp[0].weight),retain_graph=True)
            for x,y in zip(ga,gc):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-5)


if __name__=='__main__':unittest.main()
