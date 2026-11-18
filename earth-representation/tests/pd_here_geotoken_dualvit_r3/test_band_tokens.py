from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
from pd_here_geotoken_dualvit_r3.model import Layout,masked_reconstruction_loss
from pd_here_geotoken_dualvit_r3.band_tokens import BandTokenDualViT


def test_band_token_shapes_restore_and_leakage():
    torch.manual_seed(17)
    layout=Layout(('a','b'),((4,4),(2,2)),(2,2))
    model=BandTokenDualViT(layout,16,8)
    values=tuple(torch.rand(2,4,*shape,requires_grad=True) for shape in layout.native_shapes)
    qa=tuple(torch.ones_like(x,dtype=torch.bool) for x in values)
    scenarios=[
        (torch.tensor([[True,False,False,False],[False,False,True,False]]),
         torch.ones(2,2,dtype=torch.bool),'A',[[0,1],[4,5]]),
        (torch.ones(2,4,dtype=torch.bool),
         torch.tensor([[True,False],[False,True]]),'B1',[[0,2,4,6],[1,3,5,7]])]
    for keep,bands,task,expected in scenarios:
        output=model(values,qa,keep,bands)
        assert output['ids_keep'].tolist()==expected
        assert output['encoded'].shape==(2,len(expected[0]),32)
        assert output['prediction'].shape==(2,4,20)
        assert output['trace']['decoder_length']==8
        changed=tuple(torch.where(keep[:,:,None,None]&bands[:,j,None,None,None],x,
                                  torch.full_like(x,float('nan'))) for j,x in enumerate(values))
        torch.testing.assert_close(output['prediction'],model(changed,qa,keep,bands)['prediction'],rtol=0,atol=0)
        selected=torch.zeros(2,8,dtype=torch.bool).scatter(1,output['ids_keep'],True)
        torch.testing.assert_close(output['pre_position_decoder'][~selected],
                                   model.mask_token.expand(2,8,-1)[~selected],rtol=0,atol=0)
        grads=torch.autograd.grad(output['prediction'].square().mean(),values)
        for j,grad in enumerate(grads):
            assert not grad[~(keep[:,:,None,None]&bands[:,j,None,None,None]).expand_as(grad)].any()
        loss=masked_reconstruction_loss(output['prediction'],values,qa,keep,bands,task)
        assert torch.isfinite(loss)


def test_band_tokens_ragged_support_and_cross_parent():
    layout=Layout(('a','b'),((4,4),(2,2)),(2,2))
    model=BandTokenDualViT(layout,16,8)
    x=tuple(torch.rand(2,4,*shape) for shape in layout.native_shapes)
    qa=tuple(torch.ones_like(v,dtype=torch.bool) for v in x)
    for valid in qa:valid[1,3]=False
    keep=torch.tensor([[True,True,False,False],[True,False,False,False]])
    bands=torch.ones(2,2,dtype=torch.bool)
    output=model(x,qa,keep,bands)
    assert output['valid'].tolist()==[[True]*4,[True,True,False,False]]
    assert not output['prediction'][1,3].any()
    xx=[v.clone() for v in x]
    xx[0][1]+=1.
    trial=model(tuple(xx),qa,keep,bands)
    torch.testing.assert_close(output['prediction'][0],trial['prediction'][0],rtol=0,atol=0)
    for i in range(2):
        solo=model(tuple(v[i:i+1] for v in x),tuple(v[i:i+1] for v in qa),keep[i:i+1],bands[i:i+1])
        torch.testing.assert_close(output['prediction'][i:i+1],solo['prediction'],rtol=1e-5,atol=1e-6)
