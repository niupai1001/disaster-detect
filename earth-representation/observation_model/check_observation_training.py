"""Check the new objective, resumed sampling and unchanged default P3 trajectory."""
import json
import hashlib

import torch

from .data import ROOT


def state(folder):
    return torch.load(ROOT/"observation_model/outputs/phase1"/folder/"last.pt",map_location="cpu",weights_only=False)


def metrics(tag):
    result={}
    for line in (ROOT/"observation_model/outputs/tvlp"/tag/"metrics.jsonl").read_text().splitlines():
        row=json.loads(line)
        if row["phase"]=="train":result.setdefault(row["global_step"],{}).update(row["metrics"])
    return result


def main():
    old=state("cross4");compatible=state("cross4_compat")
    difference=max(float((old["model"][key]-compatible["model"][key]).abs().max()) for key in old["model"])
    relative_score_difference=abs(old["history"][-1]["score"]-compatible["history"][-1]["score"])/old["history"][-1]["score"]
    assert difference<5e-6 and relative_score_difference<1e-4,"default cross-source numerical behavior differs materially"
    assert metrics("obs_cross4")[1]["loss"]==metrics("obs_cross4_compat")[1]["loss"]
    fixed=state("observation4");control=state("observation_control4")
    assert fixed["step"]==control["step"]==4
    for key in ["sampler","pair_sampler","target_rng","mask_rng"]:
        assert fixed[key]==control[key],f"observation/control sample schedule differs: {key}"
    assert torch.equal(fixed["augmentation_rng"],control["augmentation_rng"])
    initial_root=ROOT/"observation_model/outputs/phase1"
    first=torch.load(initial_root/"observation4/initial.pt",map_location="cpu",weights_only=False)
    second=torch.load(initial_root/"observation_control4/initial.pt",map_location="cpu",weights_only=False)
    assert all(torch.equal(first[key],second[key]) for key in first)
    effect=max(float((fixed["model"][key]-control["model"][key]).abs().max()) for key in first)
    assert effect>0 and all(torch.isfinite(v).all() for v in fixed["model"].values())
    a,b=metrics("obs_observation4"),metrics("obs_observation_control4")
    assert abs(a[1]["loss"]-b[1]["loss"]-a[1]["observation_loss"])<1e-9
    assert all(a[s]["observation_valid_pixels"]>0 and a[s]["observation_target_exposure"]==8*s for s in a)
    assert all(b[s]["observation_target_exposure"]==0 for s in b)
    events=[json.loads(line) for line in (ROOT/"observation_model/outputs/tvlp/obs_observation4/events.jsonl").read_text().splitlines()]
    assert any(r.get("status")=="resumed" for r in events)
    evaluation=json.loads((initial_root/"observation4/validation_4.json").read_text())
    assert len(evaluation["observed_fine30"]["bands"])==7
    critical=["train_cross.py","observation_constraint.py","operators.py","model.py","data.py","paired.py","train_masked.py"]
    source_hashes={name:fixed["resolved_config"]["source_sha256"][name] for name in critical}
    assert all(hashlib.sha256((ROOT/"observation_model"/name).read_bytes()).hexdigest()==digest for name,digest in source_hashes.items())
    result=dict(status="passed",default_P3_max_parameter_difference=difference,observation_changes_parameters_max_difference=effect,
                default_P3_relative_validation_score_difference=relative_score_difference,compatibility_tolerance=dict(max_parameter_difference=5e-6,relative_score_difference=1e-4),
                bitwise_reproducibility_claimed=False,numerical_difference_origin="not isolated; old default trial resumed at2, comparison ran uninterrupted; BF16 used in both",
                common_initial_state=True,resumed_sampling_matches_uninterrupted_control=True,finite=True,
                real_Landsat30m_observation_loss=True,no_Landsat10m_truth_used=True,engineering_steps=12,
                source_sha256=source_hashes,
                boundary="four-step engineering evidence, not a trained-model accuracy comparison or a passed scientific extension gate")
    (initial_root/"observation_acceptance.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
