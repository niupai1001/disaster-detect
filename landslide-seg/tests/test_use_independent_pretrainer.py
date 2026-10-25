import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

scripts = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('independent_pretrainer', scripts/'pretrain_use_independent.py')
pretrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pretrain)


def test_common_initialization_and_schedule():
    reference = pretrain.initialize('physical', 'cpu').state_dict()
    for variant in pretrain.VARIANTS:
        model = pretrain.initialize(variant, 'cpu')
        for name, value in model.state_dict().items():
            if name in reference and value.shape == reference[name].shape:
                assert torch.equal(value, reference[name]), (variant, name)
    assert pretrain.learning_rate(500) == .0006
    assert pretrain.learning_rate(10000) == 0
    assert 0 < pretrain.learning_rate(1) < pretrain.learning_rate(499)


def test_target_bands_absent_from_encoder_input(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrain, 'EXTERNAL', tmp_path)
    path = tmp_path/'sample.npz'
    x = np.stack([np.full((4, 4), value, dtype='float32') for value in (1,2,3,4)])
    np.savez(path, x=x, pixel_valid=np.ones_like(x,dtype=bool), metadata=np.ones((4,3),dtype='float32'), known=np.ones((4,3),dtype=bool))
    row = dict(file=path.name, sha256=pretrain.sha(path), band_ids=['b1','b2','b3','b4'], sensor='s2', crs='EPSG:32645',transform=[10,0,0,0,-10,0])
    obs, target, valid, metadata, known, scales, trace = pretrain.batch([row], np.random.default_rng(1234), {'s2':dict(mean=[0]*4,std=[1]*4)}, 'cpu')
    assert set(trace[0]['visible_bands']).isdisjoint(trace[0]['target_bands'])
    assert set(obs['band_ids'][0]) == set(trace[0]['visible_bands'])
    assert obs['x'].shape[1] == target.shape[1] == 2
    assert set(obs) == {'x','band_ids','band_metadata','known_mask','band_valid','pixel_valid','spatial_metadata'}
    assert torch.isfinite(pretrain.masked_mae(target, target, valid))
    assert metadata.shape == (1,2,3) and known.all() and scales.eq(1).all()


def test_masked_loss_balances_samples_and_bands_and_stops_nonfinite():
    prediction = torch.tensor([[[[1.,1.]],[[3.,99.]]],[[[5.,99.]],[[99.,99.]]]],requires_grad=True)
    valid = torch.tensor([[[[True,True]],[[True,False]]],[[[True,False]],[[False,False]]]])
    target = torch.zeros_like(prediction)
    target[~valid] = float('nan')
    loss = pretrain.masked_mae(prediction,target,valid)
    assert loss.item() == 3.5
    loss.backward()
    assert (prediction.grad[~valid] == 0).all()
    target[valid] = float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        pretrain.masked_mae(prediction,target,valid)


def test_spatial_target_is_hidden_and_decoder_has_gradients(tmp_path, monkeypatch):
    monkeypatch.setattr(pretrain,'EXTERNAL',tmp_path)
    path=tmp_path/'spatial.npz'
    x=np.arange(3*8*8,dtype='float32').reshape(3,8,8)/100
    def write():
        np.savez(path,x=x,pixel_valid=np.ones_like(x,dtype=bool),metadata=np.ones((3,3),dtype='float32'),known=np.ones((3,3),dtype=bool))
    write()
    row=dict(file=path.name,sha256=pretrain.sha(path),band_ids=['b1','b2','b3'],sensor='s2',crs='EPSG:32645',transform=[10,0,0,0,-10,0])
    stats={'s2':dict(mean=[0]*3,std=[1]*3)}
    obs,target,valid,m,k=pretrain.spatial_batch([row],np.random.default_rng(1236),stats,'cpu')
    assert not (obs['pixel_valid'] & valid).any()
    assert (obs['x'][valid]==0).all()
    x[valid[0].numpy()] += 100
    write();row['sha256']=pretrain.sha(path)
    changed=pretrain.spatial_batch([row],np.random.default_rng(1236),stats,'cpu')[0]
    assert torch.equal(changed['x'],obs['x'])
    model=pretrain.initialize('spatial','cpu')
    loss=pretrain.masked_mae(model(obs,m,k,spatial=True),target,valid)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
