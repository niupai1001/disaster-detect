import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from encoder_v3.hierarchical_revised import train
from encoder_v3.hierarchical_revised.profile import synthetic
from encoder_v3.hierarchical_revised.model import Config
from encoder_v3.hierarchical_revised.tasks import Pretraining, available_tasks
from encoder_v3.shared_mae.data import EpochPlan
from encoder_v3.latent_response.checkpoint import parameter_sha

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def config():
    torch.set_num_threads(2)
    c = json.loads((ROOT/"configs/config_v3_h_revised_formal_epoch1.json").read_text(encoding="utf-8"))
    c["training"].update(device="cpu",threads=2,warmup_steps=0)
    return c


def fixture_data():
    first = synthetic(torch.device("cpu"),bands=3)
    first.identifier = "first"
    second = copy.deepcopy(first)
    second.identifier = "second"
    second.channels = second.channels[:1]
    third = copy.deepcopy(first)
    third.identifier = "third"
    samples = [first,second,third]
    rows = [dict(id=s.identifier,role=s.role,sensor="S2",bands=[{}]*len(s.channels)) for s in samples]
    return samples,SimpleNamespace(rows=rows)


def test_configuration_schedule_and_full_coverage_tail(config):
    train.validate_config(config)
    bad = copy.deepcopy(config)
    bad["formal_training_authorized"] = False
    with pytest.raises(ValueError):
        train.validate_config(bad)
    bad = copy.deepcopy(config)
    bad["decoder_variant"] = "shared_coarse"
    with pytest.raises(ValueError):
        train.validate_config(bad)
    rows = [dict(id=str(i),role="TRAIN",sensor="S2",bands=[{}]) for i in range(7442)]
    plan = EpochPlan(rows,17,32)
    assert len(plan)==233 and len(plan.batch(232))==18
    ids = [r["id"] for batch in range(len(plan)) for r in plan.batch(batch)]
    assert len(ids)==len(set(ids))==7442
    settings = dict(config["training"],warmup_steps=12)
    assert train.learning_rate(0,233,settings)==pytest.approx(1e-4/12)
    assert train.learning_rate(11,233,settings)==pytest.approx(1e-4)
    assert train.learning_rate(232,233,settings)==pytest.approx(1e-5)


def test_observation_equal_accumulation_and_task_eligibility(config,monkeypatch):
    samples,dataset=fixture_data()
    monkeypatch.setattr(train,"sample_from_dataset",lambda ds,i,resolver,device:samples[i])
    torch.manual_seed(18)
    actual=Pretraining(Config(**config["model"]))
    expected=copy.deepcopy(actual)
    optimizer=torch.optim.AdamW(actual.parameters(),lr=1e-4,weight_decay=.04)
    state=dict(cursor=0,spent_updates=0,epoch=0,observations_seen=0)
    record=train.update(actual,dataset,dataset.rows[:2],{"first":0,"second":1},None,optimizer,state,config["training"],20)
    losses=[]
    for sample in samples[:2]:
        tasks=available_tasks(sample,expected.config)
        losses.append(torch.stack([expected(sample,kind=k,seed=17,epoch=0).loss for k in tasks]).mean())
    reference=torch.stack(losses).mean()
    reference.backward()
    assert record["loss"]==pytest.approx(float(reference.detach()),rel=1e-6)
    assert record["members"][0]["eligibility"]==dict(A=True,B1=True)
    assert record["members"][1]["eligibility"]==dict(A=True,B1=False)
    for a,b in zip(actual.parameters(),expected.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=2e-4)
    bad=[dict(dataset.rows[0],role="VAL")]
    with pytest.raises(ValueError,match="Held-out"):
        train.update(actual,dataset,bad,{"first":0},None,optimizer,state,config["training"],20)


def test_identity_bound_resume_next_update(config,monkeypatch,tmp_path):
    # Bitwise CPU recovery uses one thread; real CUDA recovery checks explicit tolerances.
    torch.set_num_threads(1)
    samples,dataset=fixture_data()
    monkeypatch.setattr(train,"sample_from_dataset",lambda ds,i,resolver,device:samples[i])
    model,optimizer=train.initialize(config)
    state=dict(cursor=0,spent_updates=0,epoch=0,observations_seen=0)
    lookup={r["id"]:i for i,r in enumerate(dataset.rows)}
    train.update(model,dataset,dataset.rows,lookup,None,optimizer,state,config["training"],20)
    checkpoint=tmp_path/"checkpoint.pt"
    train.save(checkpoint,model,optimizer,state,{"config":"original"})
    train.update(model,dataset,dataset.rows,lookup,None,optimizer,state,config["training"],20)
    expected=parameter_sha(model)
    expected_values={n:p.detach().clone() for n,p in model.named_parameters()}
    other,opt=train.initialize(config)
    restored={}
    before=parameter_sha(other)
    with pytest.raises(ValueError,match="binding"):
        train.resume(checkpoint,other,opt,restored,{"config":"changed"})
    assert parameter_sha(other)==before
    train.resume(checkpoint,other,opt,restored,{"config":"original"})
    train.update(other,dataset,dataset.rows,lookup,None,opt,restored,config["training"],20)
    maximum=max(float((p.detach()-expected_values[n]).abs().max()) for n,p in other.named_parameters())
    assert parameter_sha(other)==expected and restored==state, f"maximum parameter difference: {maximum}"


@pytest.mark.parametrize("epochs", [1, 2])
def test_formal_cli_tail_validation_stop_and_completed_budget(config,monkeypatch,tmp_path,epochs):
    samples,dataset=fixture_data()
    val_samples=copy.deepcopy(samples[:2])
    for sample in val_samples:
        sample.role="VAL"
    val=SimpleNamespace(rows=[dict(r,role="VAL") for r in dataset.rows[:2]])
    monkeypatch.setattr(train,"load_data",lambda:({"TRAIN":dataset,"VAL":val},None))
    monkeypatch.setattr(train,"sample_from_dataset",
                        lambda ds,i,resolver,device:(samples if ds is dataset else val_samples)[i])
    monkeypatch.setattr(train,"binding_for",lambda *args:{"test_identity":"frozen"})
    monkeypatch.setattr(train.signal,"signal",lambda *args:None)
    release=tmp_path/"release"
    release.mkdir()
    (release/"runtime").mkdir()
    monkeypatch.setattr(train,"ROOT",release/"runtime")
    config["training"]["batch_size"]=2
    config["training"]["epochs"]=epochs
    config["viewer_root"]=str(tmp_path/"viewers")
    config["geometry_cache"]=str(tmp_path/"geometry")
    (release/"config.json").write_text(json.dumps(config))
    (release/"source_manifest.json").write_text("{}")
    (release/"preflight.json").write_text(json.dumps(dict(status="PASS",identity={"test_identity":"frozen"})))
    monkeypatch.setattr(sys,"argv",["train","--release",str(release)])
    train.main()
    run=release/"formal_run"
    state=json.loads((run/"status.json").read_text())
    updates=[json.loads(s) for s in (run/"updates.jsonl").read_text().splitlines()]
    assert state["status"]=="COMPLETE" and state["cursor"]==2*epochs and state["spent_updates"]==2*epochs
    assert state["observations_seen"]==3*epochs and state["epoch_progress"]==float(epochs)
    assert [r["batch_size"] for r in updates]==[2,1]*epochs
    assert len(list(run.glob("validation_epoch_*.json")))==epochs
    assert len(json.loads((run/"validation.json").read_text())["members"])==2
    assert state["initial_parameter_sha"]!=state["final_parameter_sha"]
    monkeypatch.setattr(sys,"argv",["train","--release",str(release),"--resume"])
    with pytest.raises(ValueError,match="completed budget"):
        train.main()



def test_continuation_preserves_weights_momenta_and_rejects_failed_gate(config,monkeypatch,tmp_path):
    from encoder_v3.latent_response.checkpoint import digest
    from encoder_v3.data import sha256
    samples,dataset=fixture_data()
    monkeypatch.setattr(train,"sample_from_dataset",lambda ds,i,resolver,device:samples[i])
    model,optimizer=train.initialize(config)
    state=dict(cursor=0,spent_updates=0,epoch=0,observations_seen=0,status="RUNNING",
               viewer={"run_id":"parent"},total_batches=1,train_observations=3)
    train.update(model,dataset,dataset.rows,{r["id"]:i for i,r in enumerate(dataset.rows)},None,
                 optimizer,state,config["training"],20)
    state.update(status="COMPLETE",final_parameter_sha=parameter_sha(model))
    binding=dict(data="data",members="members",physical_catalog="physics",model=config["model"],
                 decoder_variant="per_pixel",architecture=config["architecture"],config=digest(config),
                 code_files={"src/encoder_v3/hierarchical_revised/model.py":"same",
                             "src/encoder_v3/hierarchical_revised/train.py":"parent-runner"})
    parent=tmp_path/"parent"
    parent.mkdir()
    (parent/"config.json").write_text(json.dumps(config))
    checkpoint=parent/"formal_run/last.pt"
    train.save(checkpoint,model,optimizer,state,binding)
    gate=tmp_path/"acceptance.json"
    checks={k:True for k in ("A_improved","B1_improved","train_full_coverage","validation_full_coverage",
                             "checkpoint_binding","real_recovery_gate","tests_pass","optimizer_journal")}
    gate.write_text(json.dumps(dict(status="PASS",checks=checks,run_id="parent",final_checkpoint_sha256=sha256(checkpoint))))
    child=copy.deepcopy(config)
    child["training"]["epochs"]=10
    child["continuation"]=dict(parent_release=str(parent),checkpoint_sha256=sha256(checkpoint),
        acceptance_report=str(gate),acceptance_sha256=sha256(gate),source_run_id="parent",start_epoch=1)
    new_binding=copy.deepcopy(binding)
    new_binding["config"]=digest(child)
    new_binding["code_files"]["src/encoder_v3/hierarchical_revised/train.py"]="new-runner"
    next_model,next_opt=train.initialize(child)
    metadata=train.continue_from_checkpoint(next_model,next_opt,child,new_binding)
    assert parameter_sha(next_model)==parameter_sha(model)
    assert metadata["sampling_start_epoch"]==1 and metadata["inherited_optimizer_updates"]==1
    for key,entry in optimizer.state_dict()["state"].items():
        for name,value in entry.items():
            torch.testing.assert_close(value,next_opt.state_dict()["state"][key][name],atol=0,rtol=0)
    changed=copy.deepcopy(new_binding)
    changed["code_files"]["src/encoder_v3/hierarchical_revised/model.py"]="different-encoder"
    with pytest.raises(ValueError,match="model/data code"):
        train.continue_from_checkpoint(next_model,next_opt,child,changed)
    gate.write_text(json.dumps(dict(status="FAIL",run_id="parent",final_checkpoint_sha256=sha256(checkpoint))))
    child["continuation"]["acceptance_sha256"]=sha256(gate)
    with pytest.raises(ValueError,match="passed acceptance"):
        train.continue_from_checkpoint(next_model,next_opt,child,new_binding)