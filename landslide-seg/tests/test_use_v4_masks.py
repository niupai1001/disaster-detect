import copy
import pytest
import torch
from test_use_v4_mit import model,obs


def test_permutation_padding_invalid(model):
    o=obs(6); o['pixel_valid'][:,0]=False
    o['pixel_valid'][:,:,32:96,32:96]=False
    with torch.no_grad():
        f=model(o)
        bad=copy.deepcopy(o); bad['x'][~bad['pixel_valid']]=float('nan')
        variants=[bad,{k:v[:,1:] for k,v in o.items()}]
        order=torch.randperm(6); variants.append({k:v[:,order] for k,v in o.items()})
        for v in variants:
            actual=model(v)
            for a,b in zip([f['fine']]+f['context'],[actual['fine']]+actual['context']):
                torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
        o['pixel_valid'].zero_()
        with pytest.raises(ValueError): model(o)


def test_summary_no_leak(model):
    from pretrain_use_v4 import task_view
    o=obs(3); o['band_valid']=torch.ones(1,3,dtype=torch.bool)
    o['band_metadata']=torch.ones(1,3,3); o['known_mask']=torch.ones(1,3,3,dtype=torch.bool)
    view,hidden,summary=task_view(o,'fine',9,metadata_mode='known')
    changed=copy.deepcopy(o); changed['x'][hidden]=1e6
    view2,_,summary2=task_view(changed,'fine',9,metadata_mode='known')
    assert torch.equal(view['x'],view2['x']); assert torch.equal(summary,summary2)
    with torch.no_grad():
        torch.testing.assert_close(model(view)['context'][0],model(view2)['context'][0])


@pytest.mark.parametrize('mode',['partial','unknown'])
def test_missing_metadata(model,mode):
    from pretrain_use_v4 import task_view
    o=obs(3); o.update(band_valid=torch.ones(1,3,dtype=torch.bool),band_metadata=torch.ones(1,3,3),known_mask=torch.ones(1,3,3,dtype=torch.bool))
    view,_,_=task_view(o,'fine',12,metadata_mode=mode)
    with torch.no_grad(): f=model(view)
    assert all(torch.isfinite(x).all() for x in f['context'])
    assert not f['metadata_used'][~view['known_mask']].any()


def test_joint_metadata_permutation(model):
    o=obs(6); o['band_metadata']=torch.rand(1,6,3)+.2
    o['known_mask']=torch.ones(1,6,3,dtype=torch.bool)
    o['band_valid']=torch.ones(1,6,dtype=torch.bool)
    order=torch.randperm(6)
    with torch.no_grad():
        a=model(o); b=model({k:v[:,order] for k,v in o.items()})
    for x,y in zip([a['fine']]+a['context'],[b['fine']]+b['context']):
        torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-5)


def test_empty_sr_keys(model):
    from landslide_seg.models.use_v4_mit import masked_block
    # Last row/column fall outside floor-strided SR: there is a valid query, but no key.
    support=torch.zeros(1,33,33,dtype=torch.bool); support[:,-1,-1]=True
    tokens=torch.randn(1,33*33,32,requires_grad=True)
    out=masked_block(model.mit.block[0][0],tokens,support,33,33)
    assert torch.isfinite(out).all() and not out[~support.flatten(1)].any()
    out.sum().backward(); assert torch.isfinite(tokens.grad).all()


def test_recall_threshold_ties():
    from evaluate_use_v4 import fixed_recall
    result=fixed_recall([.9,.8,.8,.1],[1,1,0,0])
    assert result['0.5']==dict(threshold=.9,actual_recall=.5,fpr=0.)
    assert result['0.8']==dict(threshold=.8,actual_recall=1.,fpr=.5)
    assert 'unmeasured_reason' in fixed_recall([.2],[False])
