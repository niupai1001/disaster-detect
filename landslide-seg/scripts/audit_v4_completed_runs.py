"""Verify completed experiment artifacts; no inference or dataset changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    registry = read(ROOT / "configs/segformer/v4r1/experiments.json")
    protocol = read(ROOT / "manifests/v4r1/protocol.json")
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    assert protocol["frozen"] and hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest() == protocol["sha256"]
    stats_path = ROOT / "manifests/v4r1/stats_core25.json"
    stats = read(stats_path)
    fit = set(stats["fit_sample_ids"])
    assert len(registry) == 6 and len(fit) == 559 and fit == set(protocol["pools"]["25"])
    assert not fit & set(protocol["dev_ids"])
    assert all(protocol["members"][sid]["role"] == "train" for sid in fit)
    assert sha(ROOT / "manifests/split_v4.json") == protocol["legacy_split_file_sha256"]
    common_source, common_weight, runs = None, None, []
    for entry in registry:
        exp = entry["experiment"]
        run = ROOT / "outputs/segformer/s2_10band_v4r1_p20" / exp / "1234"
        summary, meta, cfg = (read(run / name) for name in ("train_summary.json", "run_meta.json", "resolved_config.json"))
        assert summary["complete"] and summary["global_step"] == 3600
        assert summary["samples_exposed"] == 230400 and summary["skipped_batches"] == 0
        assert summary["dev_events"] == {"hiroshima": 448, "thrissur": 221}
        assert summary["n_train"] == len(protocol["pools"][str(entry["pool"])])
        assert meta["stats_sha256"] == sha(stats_path)
        assert meta["protocol"]["sha256"] == protocol["sha256"]
        manifest = ROOT / cfg["segformer"]["data"]["manifest"]
        assert sha(manifest).startswith(meta["manifest"]["sha256"])
        assert meta["params"]["total"] == meta["params"]["trainable"]
        source, weight = meta["source_sha256"], meta["weight"]
        if common_source is None:
            common_source, common_weight = source, weight
        assert source == common_source and weight == common_weight
        assert [r["global_step"] for r in summary["history"]] == list(range(100, 3601, 100))
        for row in summary["history"]:
            events = row["dev"]["by_event"]
            assert set(events) == {"hiroshima", "thrissur"}
            expected = sum(e["iou"] for e in events.values()) / 2
            assert abs(expected - row["val_iou"]) < 1e-12
        checkpoint_hashes = {name: sha(run / f"{name}.pt") for name in ("best", "last")}
        for tag in ("best_val", "last_val", "best_train", "last_train", "best_common_train_reference"):
            evaluation = read(run / "eval" / f"eval_{tag}.json")
            assert not {"italy", "newzealand"} & set(evaluation["by_event"])
            pixel = evaluation["overall"]["pixel"]
            assert abs(pixel["iou"] - pixel["tp"] / (pixel["tp"] + pixel["fp"] + pixel["fn"])) < 1e-12
            assert abs(pixel["dice"] - 2 * pixel["iou"] / (1 + pixel["iou"])) < 1e-12
            assert 0 <= pixel["average_precision"] <= 1
            assert evaluation["stats_sha256"] == meta["stats_sha256"]
            assert evaluation["overall"]["negative_tiles"]["tile_false_alarm_rate"] is None
            assert checkpoint_hashes["best" if tag.startswith("best") else "last"] == evaluation["checkpoint_sha256"]
            if tag == "best_val":
                assert abs(evaluation["event_mean_iou"] - summary["best_val_iou"]) < 1e-4
        exposures = read(run / "sample_exposures.json")
        assert len(exposures) == summary["n_train"] and sum(exposures.values()) == 230400
        assert max(exposures.values()) - min(exposures.values()) <= 1
        assert (run / "eval/predictions_best_val.npz").is_file()
        runs.append({"experiment": exp, "updates": 3600, "exposures": 230400,
                     "n_train": summary["n_train"], "dev_checks": 36, "stats_sha256": meta["stats_sha256"],
                     "checkpoint_hashes_verified": True, "final_or_italy_evaluated": False})
    # Source must still match the code used by every formal training run.
    assert all(sha(ROOT / path) == digest for path, digest in common_source.items())
    report = {"status": "PASS", "n_runs": 6, "seed": 1234, "protocol_sha256": protocol["sha256"],
              "normalization_fit_train_only": True, "common_source_identity": common_source,
              "runs": runs, "scope": "artifact consistency, not a claim of label correctness or multi-seed stability"}
    out = ROOT / "reports/v4r1_results/completion_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(out)
    print("PASS: six actual runs, 216 DEV checks, exact updates/exposures, immutable input identities, no FINAL inference")


if __name__ == "__main__":
    main()
