import json
from pathlib import Path
import sys
import time
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from pd_here_geotoken_dualvit_r3.model import Layout, GeoTokenDualViT, to_subpatch, masked_reconstruction_loss
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'docs/PD_HERE_GeoToken_DualViT_R3/tools'))
from compute_budget import block,budget


def test_tensor_contracts():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    layout=Layout(('B02','B05'),((4,4),(2,2)),(2,2))
    model=GeoTokenDualViT(layout,width=16,decoder_width=8,heads=4).eval()
    values=tuple(torch.rand(2,4,*shape) for shape in layout.native_shapes)
    qa=tuple(torch.ones_like(x,dtype=torch.bool) for x in values)
    keep_a=torch.tensor([[True,False,False,False],[False,False,True,False]])
    all_bands=torch.ones(2,2,dtype=torch.bool)
    keep_b=torch.ones(2,4,dtype=torch.bool)
    vis_b=torch.tensor([[True,False],[False,True]])
    results=[]
    def check(name,fn):
        start=time.perf_counter()
        fn()
        results.append(dict(name=name,status='PASS',seconds=time.perf_counter()-start))
    out_a=model(values,qa,keep_a,all_bands)
    out_b=model(values,qa,keep_b,vis_b)
    def shape():
        assert out_a['encoded'].shape==(2,1,16)
        assert out_b['encoded'].shape==(2,4,16)
        assert out_a['prediction'].shape==(2,4,20)
        assert model.spatial_reduce.in_features==5*16
        assert model.spectral_reduce.in_features==2*16
    check('geographic_token_and_native_output_shapes',shape)
    def sequences():
        assert out_a['trace']==dict(geographic_encoder_length=1,decoder_length=4,spectral_attention_length=2,computed_band_patches=4)
        assert out_b['trace']==dict(geographic_encoder_length=4,decoder_length=4,spectral_attention_length=1,computed_band_patches=8)
    check('A_sparse_encoder_and_B1_sparse_bands',sequences)
    def leak(task):
        keep,bands,base=(keep_a,all_bands,out_a) if task=='A' else (keep_b,vis_b,out_b)
        modified=[]
        for j,x in enumerate(values):
            active=(keep & bands[:,j,None])[:,:,None,None].expand_as(x)
            modified.append(torch.where(active,x,torch.full_like(x,float('nan'))))
        trial=model(tuple(modified),qa,keep,bands)
        torch.testing.assert_close(trial['prediction'],base['prediction'],rtol=0,atol=0)
    check('A_hidden_value_and_nan_isolation',lambda:leak('A'))
    check('B1_hidden_value_and_nan_isolation',lambda:leak('B1'))
    def grad(task):
        keep,bands=(keep_a,all_bands) if task=='A' else (keep_b,vis_b)
        x=tuple(t.clone().requires_grad_() for t in values)
        model.zero_grad(set_to_none=True)
        output=model(x,qa,keep,bands)
        output['prediction'].square().mean().backward()
        visible_magnitude=0.
        for j,t in enumerate(x):
            active=(keep & bands[:,j,None])[:,:,None,None].expand_as(t)
            assert t.grad is not None and torch.equal(t.grad[~active],torch.zeros_like(t.grad[~active]))
            visible_magnitude+=t.grad[active].abs().sum().item()
        assert visible_magnitude>0
    check('A_hidden_input_gradient_zero',lambda:grad('A'))
    check('B1_hidden_input_gradient_zero',lambda:grad('B1'))
    def mask_restore():
        expected=model.mask_token[0,0].expand_as(out_a['pre_position_decoder'][~keep_a])
        torch.testing.assert_close(out_a['pre_position_decoder'][~keep_a],expected,rtol=0,atol=0)
        assert not torch.equal(out_a['decoder_input'][0,1],out_a['decoder_input'][0,2])
        assert sum(1 for name,_ in model.named_parameters() if name=='mask_token')==1
    check('one_shared_decoder_mask_and_complete_positions',mask_restore)
    def b1_mask_unused():
        model.zero_grad(set_to_none=True)
        model(values,qa,keep_b,vis_b)['prediction'].square().mean().backward()
        assert model.mask_token.grad is not None and model.mask_token.grad.abs().sum()==0
    check('B1_does_not_replace_observed_geographic_tokens',b1_mask_unused)
    def qa_isolation():
        qq=[q.clone() for q in qa];qq[0][:,:,0,0]=False
        a=model(values,tuple(qq),keep_b,all_bands)['prediction']
        vv=[v.clone() for v in values];vv[0][:,:,0,0]=float('nan')
        b=model(tuple(vv),tuple(qq),keep_b,all_bands)['prediction']
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    check('QA_invalid_value_isolation',qa_isolation)
    def loss_reduction():
        pred=torch.zeros(2,4,20)
        truth=(torch.ones(2,4,4,4),torch.full((2,4,2,2),2.))
        torch.testing.assert_close(masked_reconstruction_loss(pred,truth,qa,keep_a,all_bands,'A'),torch.tensor(2.5))
        torch.testing.assert_close(masked_reconstruction_loss(pred,truth,qa,keep_b,vis_b,'B1'),torch.tensor(2.5))
    check('pixel_band_parent_loss_weighting',loss_reduction)
    def all_branch_grads():
        model.zero_grad(set_to_none=True)
        out=model(values,qa,keep_a,all_bands)
        masked_reconstruction_loss(out['prediction'],values,qa,keep_a,all_bands,'A').backward()
        for prefix in ('spectral_block','spatial_block','spectral_reduce','spatial_reduce','fuse','global_block','prediction'):
            magnitude=sum(p.grad.abs().sum().item() for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None)
            assert magnitude>0,prefix
    check('all_required_modules_receive_A_gradients',all_branch_grads)
    def exact_flatten():
        x=torch.arange(16).reshape(1,4,4)
        expected=torch.tensor([[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]]])
        assert torch.equal(to_subpatch(x),expected)
    check('subpatch_row_major_order',exact_flatten)
    def no_obsolete():
        assert not any(isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d)) for m in model.modules())
        assert not any('descriptor' in n or 'stem' in n for n,_ in model.named_parameters())
    check('no_convolution_or_descriptor_parameters',no_obsolete)
    def no_cross_parent():
        vv=list(values);vv=[x.clone() for x in vv]
        for x in vv:x[1]+=5
        out=model(tuple(vv),qa,keep_a,all_bands)
        torch.testing.assert_close(out['prediction'][0],out_a['prediction'][0],rtol=0,atol=0)
    check('no_cross_parent_information_exchange',no_cross_parent)
    def state_roundtrip():
        other=GeoTokenDualViT(layout,16,8,4).eval();other.load_state_dict(model.state_dict())
        torch.testing.assert_close(other(values,qa,keep_a,all_bands)['prediction'],out_a['prediction'],rtol=0,atol=0)
    check('state_dict_output_roundtrip',state_roundtrip)
    def reject_empty():
        try:model(values,qa,torch.zeros_like(keep_a),all_bands)
        except ValueError:return
        raise AssertionError('empty context accepted')
    check('reject_no_visible_context',reject_empty)
    def formulas():
        assert block(16,128)==3211264
        assert budget()['spatial_flatten_projection_weight_count']==3244032
        assert budget(visible_patches=4)['decoder_tokens']==16
        assert budget(hidden_band=0)['geographic_encoder_tokens']==16
    check('MAC_and_full_decoder_length_contract',formulas)
    report=dict(status='PASS',tests=len(results),torch=torch.__version__,device='cpu',results=results,
                scope='Small synthetic tensor correctness tests only; not trained, no real imagery, no geometry loader, '
                      'no DDP/AMP/profiling validation or quality claim.',
                not_run=['real_Sentinel2_ingestion','geographic_split','GPU_throughput','DDP','BF16','training_quality','transfer'])
    destination=Path(__file__).resolve().parents[2]/'outputs/pd_here_geotoken_dualvit_r3/g0_tensor_tests.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({'tests':len(results),'status':'PASS','output':str(destination)},indent=2))

if __name__=='__main__':test_tensor_contracts()
