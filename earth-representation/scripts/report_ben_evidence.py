"""Consolidate the BigEarthNet-S2 / BigEarthNet-L physical evidence and access probes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/encoder_v3/data_download_plan"
OUT = BASE / "external_evidence"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    meta = load(BASE / "external_evidence/ben_s2_metadata.json")
    tiff = load(BASE / "external_evidence/ben_s2_tiff.json")
    benl = load(BASE / "external_evidence/ben_l_evidence.json")
    access = load(BASE / "external_access/access_evidence.json")
    calibration = load(BASE / "calibration_resolution/resolution.json")
    spots = {row["file"]: row for row in benl["samples"]}
    summary = {
        "bigearthnet_s2": {
            "patches_total": meta["patches"] + meta["snow_cloud_shadow_patches"],
            "patches_clean": meta["patches"],
            "patches_snow_cloud_shadow": meta["snow_cloud_shadow_patches"],
            "splits": meta["splits"],
            "bands": sorted(tiff),
            "grids_m": {key: tiff[key]["shape"] for key in sorted(tiff)},
            "dtype": tiff["B02"]["dtype"],
            "nodata": tiff["B02"]["nodata"],
            "in_file_scale": tiff["B02"]["scales"],
            "value_range": {key: [tiff[key]["min"], tiff[key]["max"]] for key in sorted(tiff)},
            "qa": "official patch lists only (seasonal snow / cloud-shadow), no per-pixel QA layer",
            "quantity": "sen2cor L2A surface reflectance, coded as DN/10000 per the release data card; TIFF tags carry no scale/offset",
            "verdict": "CORE_NATIVE candidate for S2 CORE10 with the limited QA tier (official lists + manual review); must not be counted as reliable-pixel QA",
        },
        "bigearthnet_l": {
            "split_rows": {"train": 269_694, "val": 123_722, "test": 125_865},
            "bands": 7, "shape": [41, 41], "grid_m": 30,
            "value_range_samples": {name: [min(b["min"] for b in row["bands"]),
                                           max(b["max"] for b in row["bands"])]
                                    for name, row in spots.items()},
            "in_file_scale": None, "in_file_nodata": None,
            "quantity": "values are consistent with Landsat surface reflectance x10000; the release ships no scale/offset or band descriptions",
            "verdict": "PENDING units: do not admit as L8_C2L2_SR7 until the Landsat-Bench data card or SSL4EO-L normalisation is documented; licence apache-2.0 on the HF card",
        },
        "calibration_pending": {
            "pending": calibration["pending"],
            "resolved_core10": calibration["resolved_core10"],
            "derived_only": calibration["resolved_derived_only"],
            "still_pending": calibration["still_pending"],
            "detail": calibration["records"],
        },
        "access": access,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    lines = [
        "# 外部物理证据与访问预审（第二阶段）", "",
        "## BigEarthNet-S2", "",
        f"- patch 总数 {summary['bigearthnet_s2']['patches_total']:,} = 清洁 {meta['patches']:,} + 雪/云/影 {meta['snow_cloud_shadow_patches']:,}。",
        f"- 官方划分：train {meta['splits']['train']:,} / val {meta['splits']['validation']:,} / test {meta['splits']['test']:,}。",
        f"- 12 个波段、原生逐波段网格：B02/03/04/08 120×120@10 m，B05/06/07/B8A/B11/B12 60×60@20 m，B01/B09 20×20@60 m；dtype uint16，nodata 0，文件内 scale 1.0、offset 0。",
        f"- 值域（20 个 patch 抽样）：B02 {tiff['B02']['min']}–{tiff['B02']['max']}，B11 {tiff['B11']['min']}–{tiff['B11']['max']}；与 sen2cor 反射率 DN/10000 一致（亮地表 >1）。",
        "- 质量证据只有官方 patch 清单，**没有逐像元 QA**；只能按“官方清单 + 人工复核”层级使用。",
        "",
        "## BigEarthNet-L", "",
        "- 7 波段 uint16、41×41@30 m、无文件内 scale/nodata/波段描述；抽样值约 7000–8900，与 Landsat 地表反射率 ×10000 一致。",
        "- 官方划分 CSV：train 269,694 / val 123,722 / test 125,865；HF 卡片 license apache-2.0。",
        "- 结论：`PENDING units`——必须拿到数据卡或 SSL4EO-L 归一化说明后才能登记 `L8_C2L2_SR7`；不能仅凭名称接入。",
        "",
        "## S2 定标 pending（10 条）", "",
        f"- 全部 10 条 canonical 记录自带 `s2:processing_baseline = 04.00`。",
        f"- 8 条原生晶格记录：缓存 4 个 10 m 波段 + 从同产品补 6 个原生 20 m 波段 → `CORE10`（TRAIN 附加记录）。",
        f"- 2 条 counterpart 记录：已是 30 m 重采样 12 通道 → 定标可追溯但登记为 `DERIVED_ONLY`，不作 CORE_NATIVE。",
        "",
        "## 公开卫星访问预审", "",
        f"- GF-2：Google Drive 文件夹可达（{access['gf2_drive'].get('entries_listed', '?')} 个条目），但没有逐对象 HTTP 入口，下载需逐文件确认令牌；仍记 `BLOCKED`。",
        f"- WorldView/GeoEye：SpaceNet 公有 S3 可达，SN3 样例包 `SN3_roads_sample.tar.gz` {764_157_199:,} 字节；可作原始产品核验入口。",
        f"- PlanetScope：公开目录只有 NICFI 月度镶嵌（`planet-nicfi-analytic/visual`），原生 PlanetScope 需账号/许可；镶嵌为派生产品。",
        f"- SPOT：WorldStrat 原始样本为 DN（无定标、CC BY-NC）；FLAIR-HUB 为 1.6 m 重采样派生产品。结论：无可用的 SPOT `CORE_NATIVE`。",
        "",
        "结论仅覆盖已抽取的原文件；未抽取部分不作合格判定。",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("BEN EVIDENCE", json.dumps({"s2_patches": summary["bigearthnet_s2"]["patches_total"],
                                     "splits": summary["bigearthnet_s2"]["splits"],
                                     "calibration": {k: summary["calibration_pending"][k]
                                                     for k in ("pending", "resolved_core10",
                                                               "derived_only", "still_pending")}},
                                    ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
