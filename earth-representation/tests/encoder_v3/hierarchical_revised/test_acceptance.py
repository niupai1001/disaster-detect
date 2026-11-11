import copy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("revised_acceptance",Path(__file__).resolve().parents[3]/"scripts/verify_v3_h_revised_acceptance.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reports():
    def report(loss):
        task=dict(loss=loss,targets=10,channel_mse={"0":loss})
        return dict(members=[dict(id="v",tasks={"A":copy.deepcopy(task),"B1":copy.deepcopy(task)})],
                    tasks={k:dict(mse=loss,eligible=1) for k in ("A","B1")})
    initial,final=report(1.),report(.5)
    update=dict(cursor=1,loss=.6,gradient_norm=2.,
                members=[dict(id="t",tasks=copy.deepcopy(final["members"][0]["tasks"]))])
    state=dict(status="COMPLETE",cursor=1,total_batches=1,spent_updates=1)
    return initial,final,[update],["t"],["v"],state,{"tests_pass":True,"source_intact":True}


def test_acceptance_requires_both_tasks_coverage_and_integrity():
    assert module.assess(*reports())["status"]=="PASS"
    for field in ("tests_pass","source_intact"):
        args=reports()
        args[-1][field]=False
        assert module.assess(*args)["status"]=="FAIL"
    args=reports()
    args[1]["members"][0]["tasks"]["B1"]["loss"]=1.1
    args[1]["tasks"]["B1"]["mse"]=1.1
    assert "B1_improved" in module.assess(*args)["failed_checks"]
    args=reports()
    args[2][0]["members"].append(copy.deepcopy(args[2][0]["members"][0]))
    assert "train_full_coverage" in module.assess(*args)["failed_checks"]
    args=reports()
    args[1]["members"][0]["tasks"]["A"]["targets"]=9
    assert "same_evaluation_target_coverage" in module.assess(*args)["failed_checks"]


def test_acceptance_rejects_nonfinite_and_uncommitted_updates():
    args=reports()
    args[2][0]["gradient_norm"]=float("nan")
    assert "finite_training" in module.assess(*args)["failed_checks"]
    args=reports()
    args[5]["spent_updates"]=2
    assert "completed_budget" in module.assess(*args)["failed_checks"]



def test_continuation_configuration_requires_exact_accepted_checkpoint(tmp_path):
    import hashlib,json
    path=Path(__file__).resolve().parents[3]/"scripts/prepare_v3_h_revised_continuation.py"
    source=importlib.util.spec_from_file_location("prepare_continuation",path)
    prepare_module=importlib.util.module_from_spec(source)
    source.loader.exec_module(prepare_module)
    parent=tmp_path/"parent"
    (parent/"formal_run").mkdir(parents=True)
    checkpoint=parent/"formal_run/last.pt"
    checkpoint.write_bytes(b"fixture")
    (parent/"source_manifest.json").write_text("{}")
    (parent/"config.json").write_text(json.dumps(dict(training={"epochs":1},authorization={})))
    (parent/"formal_run/status.json").write_text(json.dumps(dict(status="COMPLETE",viewer={"run_id":"p"})))
    gate=tmp_path/"acceptance.json"
    checks={k:True for k in ("A_improved","B1_improved","train_full_coverage","validation_full_coverage",
                             "checkpoint_binding","real_recovery_gate","tests_pass","optimizer_journal")}
    report=dict(status="PASS",checks=checks,run_id="p",
                final_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_manifest_sha256=hashlib.sha256((parent/"source_manifest.json").read_bytes()).hexdigest())
    gate.write_text(json.dumps(report))
    config=prepare_module.prepare(parent,gate)
    assert config["training"]["epochs"]==10 and config["continuation"]["start_epoch"]==1
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError,match="committed parent"):
        prepare_module.prepare(parent,gate)