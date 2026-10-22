"""Review saved artifacts and thresholds only; no model inference or training."""
import csv
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_v4r4_dev import ARMS
from summarize_frozen_decoder_probes import load, write_csv, np, plt, file_sha256

OUT = ROOT / "reports/v4r4_review"


def main():
    OUT.mkdir(exist_ok=True)
    records = [json.loads(s) for s in (ROOT / "manifests/v4r1/pool_100.jsonl").read_text(encoding="utf-8").splitlines()]
    events = {r["sample_id"]: r["event_id"] for r in records}
    scans, audit = [], []
    ref = None
    for name, relative in ARMS.items():
        run = ROOT / relative
        s, meta = load(run / "train_summary.json"), load(run / "run_meta.json")
        assert s["complete"] and s["global_step"] == 3600 and s["samples_exposed"] == 230400
        assert s["skipped_batches"] == 0 and s["n_train"] == 2223
        assert len(s["history"]) == 36
        assert max(s["history"], key=lambda h: h["val_iou"])["global_step"] == s["best_step"]
        exposure = load(run / "sample_exposures.json")
        assert len(exposure) == 2223 and sum(exposure.values()) == 230400
        assert set(exposure.values()) == {103, 104}
        hashes = {k: file_sha256(run / f"{k}.pt") for k in ("best", "last")}
        for tag in ("best_val", "last_val", "best_train", "last_train"):
            e = load(run / "eval" / f"eval_{tag}.json")
            assert e["checkpoint_sha256"] == hashes[tag.split("_")[0]]
            assert e["stats_sha256"] == meta["stats_sha256"] == file_sha256(ROOT / "manifests/v4r1/stats_core25.json")
            assert e["manifest_sha256"] == file_sha256(ROOT / "manifests/v4r1/pool_100.jsonl")
            assert not set(e["by_event"]) & {"italy", "newzealand"}
        changed = [p for p, digest in meta["source_sha256"].items()
                   if not (ROOT / p).is_file() or file_sha256(ROOT / p) != digest]
        audit.append(dict(arm=name, updates=3600, exposures=230400, best_step=s["best_step"],
                          checkpoint_hashes=hashes, current_source_differs=changed,
                          historical_reference=name == "LateDetail"))
        with np.load(run / "eval/predictions_best_val.npz") as z:
            data = {k: z[k].copy() for k in z.files}
        if ref is None:
            ref = {k: data[k] for k in ("sample_ids", "gt", "valid")}
        for key in ref:
            assert np.array_equal(ref[key], data[key])
        evs = np.array([events[str(sid)] for sid in data["sample_ids"]])
        flattened = {}
        for ev in np.unique(evs):
            selected = evs == ev
            valid = data["valid"][selected].astype(bool)
            flattened[str(ev)] = (data["probability"][selected][valid], data["gt"][selected][valid].astype(bool))
        for threshold in np.arange(.1, .901, .05):
            tp = fp = fn = 0
            row = dict(arm=name, threshold=round(float(threshold), 2))
            for ev, (p, gt) in flattened.items():
                pred = p >= np.float32(row["threshold"])
                a, b, c = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
                tp += a
                fp += b
                fn += c
                row[ev + "_iou"] = a / (a+b+c)
            row.update(event_mean_iou=np.mean([row[ev+"_iou"] for ev in flattened]),
                       precision=tp/(tp+fp), recall=tp/(tp+fn), pooled_iou=tp/(tp+fp+fn))
            if row["threshold"] == .5:
                original = load(run / "eval/eval_best_val.json")["overall"]["pixel"]
                assert (tp, fp, fn) == (original["tp"], original["fp"], original["fn"])
            scans.append(row)
    write_csv(OUT / "dev_threshold_diagnostic.csv", scans)
    (OUT / "artifact_audit.json").write_text(json.dumps(dict(status="PASS", runs=audit,
        scope="saved TRAIN/DEV artifact consistency; not label audit or independent generalization"), indent=1), encoding="utf-8")
    old_rows = list(csv.DictReader((ROOT / "reports/v4r3_results/paired_dev_objects.csv").open(encoding="utf-8-sig")))
    new_rows = list(csv.DictReader((OUT / "objects_paired_best_val.csv").open(encoding="utf-8-sig")))
    old = {(r["sample_id"], int(r["gt_object_id"])): r for r in old_rows}
    new = {(r["sample_id"], int(r["gt_id"])): r for r in new_rows if r["truncated"] == "False"}
    assert old.keys() == new.keys()
    comparisons = []
    for label, limit in (("all", 10**9), ("1_19", 19), ("1_4", 4)):
        universe = {k for k, r in old.items() if int(r["area_px"]) <= limit}
        previous = {k for k in universe if all(old[k][c] == "0" for c in
                    ("segformer_detail_hit", "segformer_detail_low_encoder_lr_hit", "dofa_upernet_pyramid_hit"))}
        current = {k for k in universe if all(new[k][c] == "False" for c in ("matched_late", "matched_self", "matched_cross"))}
        comparisons.append(dict(size=label, n=len(universe), previous_common_miss=len(previous),
                                current_common_miss=len(current), intersection=len(previous & current),
                                prior_rescued=len(previous-current), newly_common_missed=len(current-previous)))
    (OUT / "common_miss_identity_comparison.json").write_text(json.dumps(comparisons, indent=1), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in ARMS:
        sub = [r for r in scans if r["arm"] == name]
        axes[0].plot([r["threshold"] for r in sub], [r["event_mean_iou"] for r in sub], label=name)
        axes[1].plot([r["recall"] for r in sub], [r["precision"] for r in sub], marker=".", label=name)
        best = max(sub, key=lambda r: r["event_mean_iou"])
        print(name, "DEV descriptive grid maximum", json.dumps(best))
    axes[0].set(xlabel="Threshold", ylabel="DEV event mean IoU")
    axes[1].set(xlabel="Pooled pixel recall", ylabel="Pooled pixel precision")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle("Saved best checkpoints: DEV diagnostic only (no independent threshold validation)")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_diagnostic.png", dpi=150)
    plt.close(fig)
    print(json.dumps(comparisons))
    print("PASS", json.dumps(audit))


if __name__ == "__main__":
    main()
