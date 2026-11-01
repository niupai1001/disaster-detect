import numpy as np
import torch
from src.data.radiometry import reflectance
from src.models.band_stem import BandStem
from src.ssl.batching import mixed_batches

def test_known_landsat_dn():
    np.testing.assert_allclose(reflectance(np.array([10000,20000,30000,40000],np.uint16),"landsat"),[.075,.35,.625,.9],atol=1e-7)

def test_padding_does_not_enter_stem_bn():
    torch.manual_seed(0)
    a=BandStem(8).train(); b=BandStem(8).train(); b.load_state_dict(a.state_dict())
    x=torch.rand(2,6,32,32)
    y=torch.cat([x,torch.zeros(2,4,32,32)],1)
    mask=torch.tensor([[True]*6+[False]*4]*2)
    fa=a(x); fb=b(y,mask)
    torch.testing.assert_close(fa,fb[:,:6])
    fb.sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in b.parameters())

def test_batches_visit_every_item_once():
    items=[dict(image=np.full((1,8,8),i,np.float32),centre=[500],gsd=[10],bands=["B"]) for i in range(7)]
    batches=list(mixed_batches(items,3,np.random.default_rng(0)))
    assert sorted(int(x) for b in batches for x in b["image"][:,0,0,0])==list(range(7))

def test_view_major_metadata():
    from scripts.train_dino_official import BandBackbone
    b=BandBackbone(torch.nn.Identity())
    centre=torch.tensor([[500.,600.],[700.,800.]])
    b.set_meta(centre,torch.ones(2,2)*10,2,torch.ones(2,2,dtype=torch.bool))
    c,g,m=b._meta(torch.zeros(4,2,8,8))
    torch.testing.assert_close(c,centre.repeat(2,1))


def test_sensor_loss_is_two_ordinary_dino_losses_until_cross_targets_enabled():
    from scripts.train_dino_official import LocalDINOLoss
    from src.ssl.sensor_routes import observation_dino_loss
    torch.manual_seed(2)
    student=torch.randn(16,8,requires_grad=True)
    teacher=torch.randn(8,8)
    center=torch.zeros(1,8)
    chunks=student.chunk(8);targets=teacher.chunk(4)
    values=[]
    for inds,tinds in [([0,1,4,5],[0,1]),([2,3,6,7],[2,3])]:
        loss=LocalDINOLoss(8,4,.07,.07,0,1)
        values.append(loss(torch.cat([chunks[i] for i in inds]),torch.cat([targets[i] for i in tinds]),0))
    actual=observation_dino_loss(student,teacher,center,.07,cross_sensor=False)
    torch.testing.assert_close(actual,(values[0]+values[1])/2)
    paired=observation_dino_loss(student,teacher,center,.07,cross_sensor=True)
    paired.backward()
    assert student.grad.isfinite().all() and not torch.isclose(actual,paired)


def test_corrective_scale_matches_weight_mass_and_preserves_bn_buffers():
    from src.ssl.common_scale import input_support_weights,auxiliary_batch_statistics,common_scale_loss
    torch.manual_seed(0)
    x=torch.rand(2,3,32,32);lo=torch.nn.functional.avg_pool2d(x,2)
    valid=torch.ones(2,1,32,32,dtype=torch.bool)
    a=input_support_weights(x,lo,valid,(4,4),"selective",torch.Generator().manual_seed(1))
    b=input_support_weights(x,lo,valid,(4,4),"random",torch.Generator().manual_seed(1))
    torch.testing.assert_close(a.sum(),b.sum());assert (a>0).sum()==(b>0).sum()
    assert not a.requires_grad
    bn=torch.nn.BatchNorm2d(3).train();before=bn.running_mean.clone()
    with auxiliary_batch_statistics(bn):bn(x)
    torch.testing.assert_close(before,bn.running_mean)
    hi=torch.rand(2,4,8,8,requires_grad=True);low=torch.rand(2,4,4,4,requires_grad=True)
    loss,_=common_scale_loss(hi,low,x,lo,valid,"selective",torch.Generator().manual_seed(0))
    loss.backward();assert hi.grad.isfinite().all() and low.grad.isfinite().all()
