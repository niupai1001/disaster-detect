"""Read-only, CPU-only preflight for the project's current event-held-out protocol.

This checks actual manifests and provenance, never just a config's display name.
No model/data cache/checkpoint is loaded or modified. Historical official runs are
intentionally outside this current-development checker.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs/assets/research/research_policy.json"
sys.path.insert(0, str(ROOT / "src"))

from landslide_seg.data.segformer_dataset import S2_10_BANDS  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: Path, overrides: list[str]) -> dict:
    # Match current trainers' single-level _base semantics; do not silently
    # infer legacy defaults or support deeper inheritance the trainer ignores.
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    base_ref = raw.pop("_base", None)
    base = {}
    if base_ref:
        base = yaml.safe_load((path.parent / base_ref).read_text(encoding="utf-8-sig")) or {}
        require("_base" not in base, "Nested _base is not supported by current trainers")
    cfg = merge(base, raw)
    for override in overrides:
        require("=" in override, "--set requires key=value")
        key, value = override.split("=", 1)
        parts, node = key.split("."), cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node[parts[-1]] = value
    return cfg


def audit_utae_baseline(path: Path, cfg: dict, policy: dict, root: Path, policy_path: Path) -> dict:
    """U-TAE 在默认固定留出协议下的预检：成员、统计、角色与选模。"""
    require("utae_cv" in cfg and "segformer" not in cfg, "Expect a utae_cv config for the default protocol")
    s = cfg["utae_cv"]
    require(s.get("seed") == policy["seed"], "Seed differs from project policy")
    d = s["protocol"]
    directory = resolve(policy["manifest_root"], root)
    pool = resolve(d["pool"], root)
    stats = resolve(d["stats"], root)
    require(pool == directory / policy["pool_file"], "Manifest paths do not match the original held-out baseline")
    require(stats == directory / policy["stats_file"], "U-TAE baseline normalization must be stats_core25.json")
    require(d.get("roles") == {"train": policy["train_role"], "inner": policy["inner_role"]},
            "U-TAE role mapping differs from current policy")
    selection = s["eval"].get("selection")
    require(selection == policy["selection"]["utae_cv"], "Selection must use DEV event-mean IoU")
    knobs = s.get("model", {})
    positional = str(knobs.get("positional", "dates"))
    aggregation = str(knobs.get("aggregation", "attention"))
    require(positional in ("dates", "none"), "U-TAE positional must be dates/none")
    require(aggregation in ("attention", "mean"), "U-TAE aggregation must be attention/mean")
    n_frames = s.get("temporal", {}).get("n_frames")
    if n_frames is not None:
        require(type(n_frames) is int and n_frames > 0, "U-TAE n_frames 必须是正整数")
        tindex = read_json(resolve(d.get("temporal_root", "data/s2_temporal_v4r1"), root) / "index.json")
        shortest = min(len(r["dates"]) for r in tindex["rows"])
        require(shortest >= n_frames, f"NOT_ELIGIBLE: 缓存最短序列 {shortest} 帧，要求 {n_frames}")
    lock = directory / policy["protocol_file"]
    protocol = read_json(lock)
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    require(protocol.get("frozen") and protocol.get("sha256") == digest, "Frozen protocol hash mismatch")
    require(protocol.get("version", "").split("+")[0] == policy["default_protocol"],
            "Frozen protocol version mismatch")
    registered = {resolve(k, root): v for k, v in protocol["files"].items()}
    for item in (pool, stats):
        require(registered.get(item) == sha(item), f"File unregistered or modified: {item.name}")
    rows = [json.loads(line) for line in pool.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    train = [r for r in rows if r.get("split") == "train"]
    dev = [r for r in rows if r.get("split") == "val"]
    require(len(train) + len(dev) == len(rows), "Unexpected pool role")
    require({r["sample_id"] for r in train} == set(protocol["pools"]["100"]), "TRAIN membership mismatch")
    require({r["sample_id"] for r in dev} == set(protocol["dev_ids"]), "DEV membership mismatch")
    require({r["event_id"] for r in train} == set(protocol["design"]["train"])
            and {r["event_id"] for r in dev} == set(protocol["design"]["dev"]),
            "Event membership mismatch")
    norm = read_json(stats)
    fit = set(norm.get("fit_sample_ids", []))
    require(bool(fit) and fit <= {r["sample_id"] for r in train},
            "Normalization must use TRAIN only, excluding DEV/FINAL/DIAG")
    return {"ok": True, "config": str(path), "config_sha256": sha(path),
            "effective_config_sha256": hashlib.sha256(
                json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "policy_sha256": sha(policy_path), "protocol": policy["default_protocol"], "trainer": "utae_cv",
            "seed": s["seed"], "selection": selection,
            "model_knobs": {"positional": positional, "aggregation": aggregation,
                            "epochs": int(s["train"]["epochs"]),
                            "batch_size": int(s["train"]["batch_size"]),
                            "n_frames": n_frames},
            "counts": {"train": len(train), "dev": len(dev)},
            "features": {"dem": s.get("data", {}).get("dem", False),
                         "in_channels": s.get("data", {}).get("in_channels", 10)},
            "normalization_fit_n": len(fit),
            "files": {str(p.relative_to(root)): sha(p) for p in (lock, pool, stats)},
            "warnings": ["DEV is a development set, not blind TEST; FINAL/Italy diagnostic evaluation is not performed by this checker"]
                        + ([] if int(s["train"]["epochs"]) == 75 else
                           [f"本轮数 {int(s['train']['epochs'])} 与已有75轮日期组不是等轮数；只与同日程前30轮及其best比较"])}


def audit_plan(s: dict, rows: list[dict], members: dict, derived: dict | None, root: Path) -> dict:
    """逐样本帧计划：必须在派生子集中登记，成员/参考帧/期数/未来帧要与登记一致。"""
    temporal = s["temporal"]
    require(derived is not None and derived.get("plans"), "帧计划必须登记在派生子集协议中")
    path = resolve(temporal["plan"], root)
    registered = {resolve(v["path"], root): v for v in derived["plans"].values()}
    require(path in registered, "帧计划未登记在派生子集协议中")
    meta = registered[path]
    plan = read_json(path)
    require(set(plan) == {r["sample_id"] for r in rows}, "帧计划的成员与清单不一致")
    require(int(meta["n_samples"]) == len(plan), "帧计划登记的样本数与文件不一致")
    future = 0
    for row in rows:
        sid = row["sample_id"]
        names = [int(i) for i in plan[sid]]
        dates = members[sid]["dates"]
        anchor = int(row["time_index"])
        require(len(set(names)) == len(names), f"{sid} 帧计划含重复帧")
        require(all(0 <= i < len(dates) for i in names), f"{sid} 帧计划帧下标越界")
        require(anchor in names, f"{sid} 帧计划未包含参考帧")
        if temporal.get("n_frames") is not None:
            require(len(names) == int(temporal["n_frames"]),
                    f"{sid} 帧计划长度与声明的 n_frames 不一致")
        future += sum(1 for i in names if dates[i] > row["date"])
    require(future == int(meta.get("n_future_frames", -1)), "帧计划的未来帧数与登记不一致")
    require(future == 0 or str(temporal.get("mode")) == "multi",
            "含参考日期之后的帧时必须显式声明 mode=multi（回顾性条件）")
    return {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha(path),
            "n_samples": len(plan), "n_frames": sorted({len(v) for v in plan.values()}),
            "future_frames": future,
            "max_days_after_anchor": meta.get("max_days_after_anchor")}


def audit_temporal(data: dict, s: dict, rows: list[dict], root: Path,
                   derived: dict | None = None) -> dict:
    """只读核对时间缓存：成员、参考帧日期、帧数、quality 侧表哈希、汇聚开关。"""
    tdir = resolve(data["temporal_root"], root)
    index_path = tdir / "index.json"
    require(index_path.is_file(), f"Temporal cache index missing: {index_path}")
    index = read_json(index_path)
    members = {r["sample_id"]: r for r in index.get("rows", [])}
    require(bool(members), "Temporal cache index has no rows")
    missing = [r["sample_id"] for r in rows if r["sample_id"] not in members]
    require(not missing, f"Temporal cache lacks {len(missing)} samples, e.g. {missing[:3]}")
    for row in rows:
        rec = members[row["sample_id"]]
        dates = rec["dates"]
        require(len(dates) >= 1, f"Empty date list: {row['sample_id']}")
        require(0 <= int(row["time_index"]) < len(dates), f"time_index out of range: {row['sample_id']}")
        require(dates[int(row["time_index"])] == row["date"],
                f"Reference frame date mismatch: {row['sample_id']}")
    mode = str(s.get("temporal", {}).get("mode", "single"))
    aggregation = str(s.get("temporal", {}).get("aggregation", "identity"))
    require(mode in ("single", "multi", "history"), f"Unknown temporal mode: {mode!r}")
    require(aggregation in ("identity", "mean", "content", "date"),
            f"Unknown temporal aggregation: {aggregation!r}")
    require((mode == "single") == (aggregation == "identity"),
            "single must use aggregation=identity; multi/history must use mean/content/date")
    count = s.get("temporal", {}).get("n_frames")
    window = s.get("temporal", {}).get("window_days")
    if window is not None:
        require(mode == "history", "window_days 只适用于 history（截止参考日期）")
        require(type(window) is int and window > 0, "window_days must be a positive integer")
    plan_info = None
    if s.get("temporal", {}).get("plan"):
        require(mode != "single", "帧计划不能与单期模式混用")
        plan_info = audit_plan(s, rows, members, derived, root)
    elif count is not None:
        require(type(count) is int and count > 0, "n_frames must be a positive integer")
        for row in rows:
            dates = members[row["sample_id"]]["dates"]
            if mode == "single":
                available = 1
            elif mode == "history":
                anchor = np.datetime64(row["date"], "D").astype(int)
                days = np.asarray([np.datetime64(d, "D").astype(int) for d in dates])
                past = days <= anchor
                if window:
                    past &= days >= anchor - int(window)
                available = int(past.sum())
            else:
                available = len(dates)
            require(available >= count, f"NOT_ELIGIBLE: {row['sample_id']} has {available} frames, needs {count}")
    qinfo = None
    quality = data.get("quality")
    if quality:
        qdir = resolve(quality, root)
        complete = qdir / "complete.json"
        if complete.is_file():
            qmeta = read_json(complete)
            require(qmeta.get("index_sha256") == sha(index_path),
                    "Quality sidecar index hash mismatch")
            qinfo = {"path": str(qdir), "enabled": True,
                     "complete_sha256": sha(complete)}
        else:
            qinfo = {"path": str(qdir), "enabled": False,
                     "note": "No complete.json; missing-value filling disabled"}
    n_frames = sorted({len(r["dates"]) for r in index["rows"]})
    return {"root": str(tdir), "index_sha256": sha(index_path),
            "n_index_rows": len(index["rows"]), "frames_per_sample": n_frames,
            "mode": mode, "aggregation": aggregation, "requested_n_frames": count,
            "window_days": window, "plan": plan_info, "quality": qinfo}


def audit_budget(settings: dict) -> dict | None:
    """更新制预算的只读解析：停止步数、日程长度、预热与预定 checkpoint。"""
    if str(settings.get("train", {}).get("budget_mode")) != "updates":
        return None
    from landslide_seg.training.budget import resolve_update_budget

    budget = resolve_update_budget(settings["train"])
    return {**budget.as_dict(),
            "warmup_updates": budget.warmup_updates(float(settings["optim"]["warmup_frac"])),
            "encoder_lr": float(settings["optim"]["encoder_lr"]),
            "head_lr": float(settings["optim"]["head_lr"])}


def audit_baseline(path: Path, cfg: dict, policy: dict, root: Path, policy_path: Path) -> dict:
    require("segformer" in cfg, "Original held-out baseline config required; adapt other trainer interfaces first")
    s = cfg["segformer"]
    require(s.get("seed") == policy["seed"], "Seed differs from project policy")
    data = s["data"]
    directory = resolve(policy["manifest_root"], root)
    pool = resolve(data["manifest"], root)
    stats = resolve(data["stats"], root)
    subset = data.get("subset_protocol")
    derived = read_json(resolve(subset, root)) if subset else None
    if derived is not None:
        # 派生清单必须显式登记基准清单、成员与哈希；不能靠改 version 冒充全体
        require(resolve(derived["base_manifest"], root) == directory / policy["pool_file"],
                "Derived manifest does not derive from the original held-out pool")
        require(resolve(derived["manifest"], root) == pool, "Configured manifest differs from the registered subset")
        require(sha(pool) == derived["manifest_sha256"], "Derived manifest hash differs from its registration")
        require(int(derived["min_frames_at_or_before_reference"]) >= 1,
                "Derived subset must declare its eligibility cutoff")
    else:
        require(pool == directory / policy["pool_file"], "Manifest paths do not match the original held-out baseline")
    require(bool(data.get("protocol")), "Explicit frozen data.protocol is required")
    lock = resolve(data["protocol"], root)
    require(stats.parent == directory and lock.parent == directory, "Statistics/protocol must belong to original held-out baseline")
    require(int(data.get("pool_fraction", 0)) == 100, "Use original full 2223-sample TRAIN baseline")
    selection = s["eval"].get("selection_metric")
    require(selection == policy["selection"]["segformer"], "Selection must use DEV event-mean IoU")
    protocol = read_json(lock)
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(protocol.get("frozen") and protocol.get("sha256") == digest, "Frozen protocol hash mismatch")
    require(protocol.get("version", "").split("+")[0] == policy["default_protocol"], "Frozen protocol version mismatch")
    registered = {resolve(k, root): v for k, v in protocol["files"].items()}
    # 派生清单不在基准协议的登记表里，它的身份由上一步的子集登记核对
    for item in ((stats,) if derived is not None else (pool, stats)):
        require(registered.get(item) == sha(item), f"File unregistered or modified: {item.name}")
    rows = [json.loads(line) for line in pool.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    train = [r for r in rows if r.get("protocol_role") == policy["train_role"]]
    dev = [r for r in rows if r.get("protocol_role") == policy["inner_role"]]
    require(len(train) + len(dev) == len(rows), "Unexpected pool role")
    require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate/overlapping sample identities")
    train_ids = {r["sample_id"] for r in train}
    if derived is not None:
        require(train_ids == set(derived["train_ids"]), "TRAIN membership differs from the registered subset")
        require(len(train) == int(derived["counts"]["train"]), "Subset TRAIN count differs from its registration")
        require(train_ids <= set(protocol["pools"]["100"]), "Derived subset is not a subset of the original TRAIN pool")
        require({r["sample_id"] for r in dev} == set(derived["dev_ids"]), "DEV membership differs from the registered subset")
    else:
        require(train_ids == set(protocol["pools"]["100"]), "TRAIN membership mismatch")
    require({r["sample_id"] for r in dev} == set(protocol["dev_ids"]), "DEV membership mismatch")
    train_events, dev_events = ({r["event_id"] for r in group} for group in (train, dev))
    require(dev_events == set(protocol["design"]["dev"]), "Original event membership mismatch")
    require(train_events <= set(protocol["design"]["train"]),
            "Denied: derived TRAIN events must stay inside the original TRAIN design")
    if derived is None:
        require(train_events == set(protocol["design"]["train"]), "Original event membership mismatch")
    require(not train_events & dev_events, "TRAIN/DEV events overlap")
    for row in rows:
        member = protocol["members"].get(row["sample_id"], {})
        for field in ("event_id", "cache", "cache_sha256", "date", "time_index"):
            require(row.get(field) == member.get(field), f"Frozen sample identity changed: {row['sample_id']}/{field}")
        require(row.get("split") == member.get("role"), "Sample role differs from frozen protocol")
    norm = read_json(stats)
    fit = set(norm.get("fit_sample_ids", []))
    # 派生子集沿用 559 块 TRAIN 核心统计；只要求拟合样本属于原 TRAIN，不要求都属于该子集
    require(bool(fit) and fit <= (set(protocol["pools"]["100"]) if derived is not None else train_ids),
            "Normalization must use TRAIN only, excluding DEV/FINAL/DIAG")
    require(set(norm.get("fit_events", {})) == {protocol["members"][sid]["event_id"] for sid in fit}, "Normalization fit event mismatch")
    reference_path = directory / policy["stats_file"]
    require(registered.get(reference_path) == sha(reference_path), "Original common-core normalization hash mismatch")
    reference = read_json(reference_path)
    require(set(reference["fit_sample_ids"]) == set(protocol["pools"]["25"]), "Original common-core membership mismatch")
    require(norm["mean"][:10] == reference["mean"] and norm["std"][:10] == reference["std"], "Original S2 normalization changed; declare a separate data experiment")
    n_channels = int(s.get("model", {}).get("in_chans", 10))
    require(len(norm["mean"]) == len(norm["std"]) == n_channels, "Normalization/input channel mismatch")

    # 光谱机制预检：新增指数通道的统计、关系/门控的身份与融合声明必须明确，
    # 不能用未登记的统计或含 DEV 的拟合。这里只读配置与 JSON，不加载模型。
    model_cfg = s.get("model", {}) or {}
    model_type = str(model_cfg.get("type", "segformer"))
    spectral = None
    extra_files = {}
    if model_type == 'use':
        tc = s['train']
        require(tc.get('batch_size') == 8 and tc.get('freeze_bn_running_stats') is True,
                'USE首轮要求物理batch8并冻结BN运行统计')
        require(s['model'].get('backbone') == 'b0' and s['model'].get('stem') is True,
                'USE首轮固定B0与原网格细节')
        require(model_cfg.get('adapter','direct10')=='direct10' and model_cfg.get('extra_channel','none')=='none'
                and model_cfg.get('refine','none')=='none' and model_cfg.get('exchange','none')=='none',
                'USE首轮不叠加地形、投影、精分或早期交换')
        require(not s.get('temporal') and not s.get('optim',{}).get('first_conv_lr'), 'USE首轮固定单期及统一编码器学习率')
        require(isinstance(data.get('mixed_bands',False),bool), 'mixed_bands必须是布尔值')
        require(s['loss']['bce_weight'] == .5 and s['loss']['dice_weight'] == .5,
                'USE首轮固定监督损失')
        use = model_cfg.get('use', {})
        require(all(isinstance(use.get(k,True),bool) for k in ('attention','metadata')), 'USE开关必须是布尔值')
        require(use.get('queries',4) in (0,4) and use.get('hidden',32) in (32,365), 'USE首轮只使用已登记查询数与隐藏宽度')
        require(use.get('width', 32) == 32 and use.get('heads', 4) == 4
                and use.get('relation_grid', 2) == 2, 'USE首轮结构维度不符')
        require(use.get('identity', 'physical') in ('physical', 'random'), '非法元数据身份')
        if model_cfg.get('use_pretrained'):
            pretraining=root/'outputs/use/pretraining/summary.json'
            require(pretraining.is_file(), 'USE预训练尚未完成验收')
            summary=read_json(pretraining);weights=resolve(model_cfg['use_pretrained'],root)
            require(summary.get('complete') is True and summary.get('updates')==10000, 'USE预训练预算未完成')
            require(sha(weights)==model_cfg.get('use_pretrained_sha256')==summary.get('best_sha256'), 'USE预训练权重来源或哈希不符')
            require(sha(root/'outputs/use/external/manifest.json')==summary.get('manifest_sha256'), '预训练成员清单发生改变')
        quality_path = root/'outputs/use/quality/index.json'
        require(quality_path.is_file(), '先生成独立输入质量侧表')
        quality = read_json(quality_path)
        for rec in rows:
            row = quality.get(rec['sample_id'], {})
            require(row.get('cache_sha256') == rec['cache_sha256'] and row.get('time_index') == rec['time_index'],
                    '输入质量侧表成员或观测身份不符')
        extra_files[str(quality_path)] = sha(quality_path)
        spectral = {'level': 'use', 'config': use, 'mixed_bands': bool(data.get('mixed_bands', False)),
                    'quality_index_sha256': sha(quality_path)}
    if model_type == "spectral_ladder":
        sp = model_cfg.get("spectral_ladder", {}) or {}
        level, variant = str(sp.get("level", "S1")), str(sp.get("variant", "treatment"))
        extra_path = sp.get("extra_stats")
        if level == "S1":
            if variant == "treatment":
                require(bool(extra_path), "S1 treatment 必须登记四个指数通道的 TRAIN 统计")
                ex_path = resolve(extra_path, root)
                require(ex_path.is_file(), f"新增通道统计文件不存在: {ex_path}")
                ex = read_json(ex_path)
                require(len(ex["mean"]) == len(ex["std"]) == 4, "新增通道统计必须是四个指数")
                require(ex.get("channels") == ["NDVI", "NDMI", "NBR", "REND"],
                        "新增通道顺序必须是 NDVI/NDMI/NBR/REND")
                require(ex.get("bands") == list(S2_10_BANDS), "新增通道统计的十波段身份不符")
                ex_fit = set(ex["fit_sample_ids"])
                require(bool(ex_fit) and ex_fit <= train_ids,
                        "新增通道统计必须只用 TRAIN 拟合")
                require(set(ex.get("fit_events", {})) == {protocol["members"][sid]["event_id"] for sid in ex_fit},
                        "新增通道统计的拟合事件与成员不符")
                require(float(ex.get("eps", 0)) == 1e-4, "指数 eps 必须与契约一致")
                require(ex.get("radiometry") == "harmonized_dn_v1", "指数必须用 harmonized 反射率")
                require(sp.get("radiometry") == "harmonized_dn_v1", "S1 训练也必须用 harmonized 反射率")
                extra_files[str(ex_path.relative_to(root))] = sha(ex_path)
                spectral = {"level": "S1", "variant": variant,
                            "extra_stats": str(ex_path.relative_to(root)).replace("\\", "/"),
                            "extra_channels": list(ex["channels"]),
                            "extra_mean": [float(v) for v in ex["mean"]],
                            "extra_std": [float(v) for v in ex["std"]],
                            "first_layer": "新增四列在 direct10 首层零初始化；主干预训练列不变"}
            else:
                require(not extra_path, "S1 控制组不应套用指数统计；原始通道已由 data.stats 标准化")
                spectral = {"level": "S1", "variant": variant,
                            "extra_channels": ["B04", "B08", "B8A", "B11"],
                            "extra_normalization": "data.stats 十波段 z-score（dataset 完成）"}
        else:
            require(not extra_path, "只有 S1 允许新增通道统计")
    elif model_type in ("spectral_relation", "spectral_gate"):
        key = "spectral_relation" if model_type == "spectral_relation" else "spectral_gate"
        sp = model_cfg.get(key, {}) or {}
        identity = str(sp.get("identity", "wavelength"))
        relation = str(sp.get("relation_mode", "measured"))
        width, heads, layers = int(sp.get("width", 64)), int(sp.get("heads", 4)), int(sp.get("layers", 2))
        patch, image_size = int(sp.get("patch", 4)), int(data.get("image_size", 128))
        require(identity in ("wavelength", "channel"), "identity 必须是 wavelength/channel")
        require(relation in ("measured", "none", "global"), "relation_mode 必须是 measured/none/global")
        require(width % heads == 0, "width 必须能被 heads 整除")
        require(layers >= 1 and patch >= 1 and image_size % patch == 0,
                "关系块层数/分块必须覆盖完整影像")
        require(sp.get("extra_stats") is None, "关系/门控组不接受 S1 额外通道统计")
        spectral = {"level": model_type, "identity": identity, "relation_mode": relation,
                    "width": width, "heads": heads, "layers": layers, "patch": patch}
        if model_type == "spectral_gate":
            fusion = str(sp.get("fusion", "learn"))
            require(fusion in ("add", "fixed", "learn"), "gate fusion 必须是 add/fixed/learn")
            spectral["fusion"] = fusion
            spectral["fixed_gate"] = float(sp.get("fixed_gate", 0.5))

    temporal = None
    if data.get("temporal_root"):
        temporal = audit_temporal(data, s, rows, root, derived)
    budget = audit_budget(s)
    warnings = ["DEV is a development set, not blind TEST; FINAL/Italy diagnostic evaluation is not performed by this checker"]
    if budget and budget["stop_updates"] != budget["schedule_updates"]:
        warnings.append(
            f"停止步数 {budget['stop_updates']} 短于日程长度 {budget['schedule_updates']}："
            "只与同日程前缀比较，不能与历史全程运行当等预算")
    if derived is not None:
        warnings.append("派生共同子集：只与同成员配对组比较，不与原 2223 块全体混排")
    return {"ok": True, "config": str(path), "config_sha256": sha(path),
            "effective_config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "policy_sha256": sha(policy_path), "protocol": policy["default_protocol"], "fold": None,
            "seed": s["seed"], "selection": selection, "counts": {"train": len(train), "dev": len(dev)},
            "events": {"train": sorted(train_events), "dev": sorted(dev_events)},
            "normalization_fit_n": len(fit),
            "files": {**{str(p.relative_to(root)): sha(p) for p in (lock, pool, stats)},
                      **extra_files},
            "spectral": spectral,
            "subset": None if derived is None else {
                "protocol": str(resolve(subset, root).relative_to(root)).replace("\\", "/"),
                "rule": derived["rule"],
                "min_frames_at_or_before_reference": derived["min_frames_at_or_before_reference"],
                "base_manifest": derived["base_manifest"],
                "manifest_sha256": derived["manifest_sha256"]},
            "budget": budget,
            "temporal": temporal,
            "warnings": warnings}


def audit_config(config: str | Path, overrides: list[str] | None = None,
                 root: Path = ROOT, policy_path: Path | None = None,
                 protocol: str | None = None) -> dict:
    root = root.resolve()
    policy_path = policy_path or root / "docs/assets/research/research_policy.json"
    policy = read_json(policy_path)
    path = resolve(str(config), root)
    cfg = load_config(path, overrides or [])
    selected = protocol or policy["default_protocol"]
    if selected == policy["default_protocol"]:
        if "utae_cv" in cfg:
            return audit_utae_baseline(path, cfg, policy, root, policy_path)
        return audit_baseline(path, cfg, policy, root, policy_path)
    require(selected in policy.get("optional_protocols", {}), "Unknown explicit historical protocol")
    policy = merge(policy, policy["optional_protocols"][selected])
    policy["default_protocol"] = selected
    kind = "utae_cv" if "utae_cv" in cfg else "segformer"
    require(kind in cfg, "Expected explicit segformer or utae_cv config")
    settings = cfg[kind]
    require(settings.get("seed") == policy["seed"], "Seed differs from project policy")
    if kind == "utae_cv":
        data = settings["protocol"]
        pool, outer, stats = (resolve(data[k], root) for k in ("pool", "outer", "stats"))
        fold = settings.get("fold")
        require(data.get("roles") == {"train": policy["train_role"], "inner": policy["inner_role"]},
                "U-TAE role mapping differs from current policy")
        selection = settings["eval"].get("selection")
        protocol_path = pool.parent / "protocol.json"
    else:
        data = settings["data"]
        pool, stats = (resolve(data[k], root) for k in ("manifest", "stats"))
        fold = pool.parent.name
        outer = pool.parent / policy["outer_file"]
        selection = settings["eval"].get("selection_metric")
        require(bool(data.get("protocol")), "Explicit frozen data.protocol is required")
        protocol_path = resolve(data["protocol"], root)
        require(int(data.get("pool_fraction", 0)) == 100, "Current formal comparison requires full TRAIN pool")
    require(fold in policy["folds"], "Not a current event fold; official/historical split is not the default")
    fold_dir = resolve(policy["manifest_root"], root) / fold
    require(pool == fold_dir / policy["pool_file"] and outer == fold_dir / policy["outer_file"],
            "Manifest paths do not match the current event protocol")
    require(stats.parent == fold_dir and protocol_path.parent == fold_dir,
            "Statistics/protocol must belong to this same fold")
    require(selection == policy["selection"][kind], "Selection must use INNER event-mean IoU")
    protocol = read_json(protocol_path)
    payload = {k: v for k, v in protocol.items() if k != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    require(protocol.get("frozen") and protocol.get("sha256") == digest, "Frozen protocol hash mismatch")
    require(protocol.get("version", "").split("+")[0] == f"{policy['default_protocol']}/{fold}",
            "Frozen protocol version mismatch")
    registered = {resolve(k, root): v for k, v in protocol["files"].items()}
    for item in (pool, outer):
        require(registered.get(item) == sha(item), f"Manifest is unregistered or modified: {item.name}")
    rows = lambda p: [json.loads(line) for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    pool_rows, outer_rows = rows(pool), rows(outer)
    train = [r for r in pool_rows if r.get("protocol_role") == policy["train_role"]]
    inner = [r for r in pool_rows if r.get("protocol_role") == policy["inner_role"]]
    require(len(train) + len(inner) == len(pool_rows), "Unexpected pool role")
    require(all(r.get("protocol_role") == policy["outer_role"] for r in outer_rows), "Unexpected outer role")
    all_rows = pool_rows + outer_rows
    require(len({r["sample_id"] for r in all_rows}) == len(all_rows), "Duplicate/overlapping sample identities")
    groups = (train, inner, outer_rows)
    events = [{r["event_id"] for r in group} for group in groups]
    require(all(groups) and not any(events[i] & events[j] for i, j in ((0, 1), (0, 2), (1, 2))),
            "Empty role or event overlap")
    for group, key in zip(groups, ("train_events", "inner", "outer")):
        require({r["event_id"] for r in group} == set(protocol["design"][key]), f"Event design mismatch: {key}")
    require({r["sample_id"] for r in train} == set(protocol["pools"]["100"]), "TRAIN membership mismatch")
    require({r["sample_id"] for r in inner} == set(protocol["dev_ids"]), "INNER membership mismatch")
    require({r["sample_id"] for r in outer_rows} == set(protocol["test_ids"]), "OUTER membership mismatch")
    for row in all_rows:
        member = protocol["members"].get(row["sample_id"], {})
        for field in ("event_id", "cache", "cache_sha256", "date", "time_index"):
            require(row.get(field) == member.get(field), f"Frozen sample identity changed: {row['sample_id']}/{field}")
        require(row.get("split") == member.get("role"), "Sample role differs from frozen protocol")
    norm = read_json(stats)
    fit = set(norm.get("fit_sample_ids", []))
    require(fit == {r["sample_id"] for r in train}, "Normalization provenance is not this complete TRAIN pool")
    require(norm.get("fold") == fold and set(norm.get("fit_events", {})) == events[0], "Normalization fold/events mismatch")
    n_channels = int(settings["data"].get("in_channels", 11)) if kind == "utae_cv" else int(settings.get("model", {}).get("in_chans", 10))
    require(len(norm["mean"]) == len(norm["std"]) == n_channels, "Normalization/input channel mismatch")
    # Derived U-TAE stats_dem is not registered in the base S2-only protocol.
    # Its provenance and shared ten S2 statistics can still be checked, but do
    # not claim it has a frozen protocol hash registration that does not exist.
    warnings = []
    if stats in registered:
        require(registered[stats] == sha(stats), "Registered statistics hash mismatch")
    else:
        require(kind == "utae_cv" and stats.name == "stats_dem.json", "Unregistered statistics")
        reference_path = fold_dir / "stats_train.json"
        require(registered.get(reference_path) == sha(reference_path), "Base S2 statistics hash mismatch")
        reference = read_json(reference_path)
        require(norm["mean"][:10] == reference["mean"] and norm["std"][:10] == reference["std"], "Derived stats changed the shared S2 normalization")
        warnings.append("stats_dem provenance checked; DEM values are not registered in the base protocol hash and were not refitted here")
    return {"ok": True, "config": str(path), "config_sha256": sha(path),
            "effective_config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "policy_sha256": sha(policy_path), "protocol": policy["default_protocol"], "fold": fold,
            "seed": settings["seed"], "selection": selection,
            "counts": dict(zip(("train", "inner", "outer"), map(len, groups))),
            "events": dict(zip(("train", "inner", "outer"), map(sorted, events))),
            "files": {str(p.relative_to(root)): sha(p) for p in (protocol_path, pool, outer, stats)},
            "warnings": warnings}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--protocol", default=None, help="默认原固定事件留出；EVENT_CV_V5仅用于显式历史四折复查")
    args = ap.parse_args()
    results = []
    for config in args.config:
        try:
            results.append(audit_config(config, args.set, protocol=args.protocol))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            results.append({"ok": False, "config": config, "error": str(exc)})
    print(json.dumps(results, ensure_ascii=False, indent=1))
    raise SystemExit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
