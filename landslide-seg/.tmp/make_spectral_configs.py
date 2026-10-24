from pathlib import Path

ROOT = Path(r"N:\Users\yzh\Disater\landslide-seg")
OUT = ROOT / "configs" / "segformer" / "v4r2_spectral"
OUT.mkdir(parents=True, exist_ok=True)

COMMON = """\
segformer:
  experiment: {experiment}
  seed: 1234
  deterministic: true
  output_dir: outputs/segformer
  data:
    version: v4r2_spectral
    source_root: data/sen12
    cache_dir: data/s2_10band_v4_p20
    manifest: manifests/v4r1/pool_100.jsonl
    quarantine: manifests/s2_10band_v4_p20_quarantine.jsonl
    stats: manifests/v4r1/stats_core25.json
    test_event: italy
    val_event: hiroshima
    max_cloud: 0.5
    min_valid_ratio: 0.5
    max_label_invalid_ratio: 0.6
    warn_days_after_event: 365
    post_event_window_days: 120
    image_size: 128
    num_workers: 0
    preload: true
    pool_fraction: 100
    protocol: manifests/v4r1/protocol.json
  model:
    type: {mtype}
    backbone: b0
    adapter: direct10
    stem: true
    refine: none
    in_chans: 10
    band_order:
    - B02
    - B03
    - B04
    - B05
    - B06
    - B07
    - B08
    - B8A
    - B11
    - B12
    extra_channel: none
{spec}    pretrained_dir: weights/segformer/segformer-b0-finetuned-ade-512-512
    pretrained_source: nvidia/segformer-b0-finetuned-ade-512-512
  loss:
    bce_weight: 0.5
    dice_weight: 0.5
  optim:
    encoder_lr: 6.0e-05
    head_lr: 0.0006
    weight_decay: 0.01
    grad_clip: 1.0
    warmup_frac: 0.05
    schedule: poly
    poly_power: 1.0
  train:
    epochs: 150
    batch_size: 64
    amp: false
    amp_dtype: bf16
    log_every: 20
    max_steps: 2000
    stop_updates: 2000
    schedule_updates: 3600
    budget_mode: updates
    eval_every_updates: 100
    report_steps:
    - 1000
    - 1500
    - 2000
  eval:
    threshold: 0.5
    scale_thresholds_ha:
    - 0.5
    - 2.0
    object_iou_threshold: 0.25
    object_iou_strict: 0.5
    boundary_tol_px: 1
    selection_metric: dev_event_mean_iou
"""

BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def spec_block(lines):
    return "".join(f"    {line}\n" for line in lines)


def write(name, header, experiment, mtype, spec_lines):
    text = "".join(f"# {line}\n" for line in header)
    text += COMMON.format(experiment=experiment, mtype=mtype, spec=spec_block(spec_lines))
    # preserve the requested band order explicitly for readers
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", name)


write(
    "s1_indices.yaml",
    [
        "S1：十波段后追加 NDVI/NDMI/NBR/REND 四个指数（共 14 通道）。",
        "新增四个指数通道用 manifests/v4r1/stats_s1_indices4.json 的 TRAIN 统计独立标准化；",
        "十波段仍用 stats_core25.json。指数在 harmonized 反射率上计算，eps=1e-4。",
        "首层：direct10 把 RGB 权重拷到 B04/B03/B02，其余列（含新增四列）零初始化；",
        "处理组与对照组的主干预训练列完全相同，不重置主干。",
        "预算：停止 2000 次成功更新，学习率日程沿用旧 3600 步（预热 180）。",
    ],
    "SegFormer-B0-S1Indices-14ch",
    "spectral_ladder",
    [
        "spectral_ladder:",
        "  level: S1",
        "  variant: treatment",
        "  rank: 4",
        "  radiometry: harmonized_dn_v1",
        "  extra_stats: manifests/v4r1/stats_s1_indices4.json",
    ],
)
write(
    "s1_control.yaml",
    [
        "S1 容量对照：同样追加四个通道，取 B04/B08/B8A/B11 的原始 z-score，不含物理公式。",
        "四个对照通道已由 dataset 用 TRAIN 十波段统计标准化，不再套用指数统计。",
        "通道数、首层参数与初始化（新增列零初始化）与处理组一致。",
        "预算：停止 2000 次成功更新，学习率日程沿用旧 3600 步（预热 180）。",
    ],
    "SegFormer-B0-S1Control-14ch",
    "spectral_ladder",
    ["spectral_ladder:", "  level: S1", "  variant: control", "  rank: 4"],
)

REL = [
    "spectral_relation:",
    "  patch: 4",
    "  width: 64",
    "  heads: 4",
    "  layers: 2",
]
write(
    "s3_a_linear.yaml",
    [
        "S3-A 通道混合控制：逐像元满秩 10×10 线性混合，x + up(down(x))，up 零初始化。",
        "两层 1×1 卷积之间没有激活，只是普通通道混合；不代表注意力或乘性关系。",
        "预算：停止 2000 次成功更新，学习率日程沿用旧 3600 步（预热 180）。",
    ],
    "SegFormer-B0-S3A-LinearMix",
    "spectral_ladder",
    ["spectral_ladder:", "  level: S3", "  variant: control", "  rank: 4"],
)
write(
    "s3_b_idattn.yaml",
    [
        "S3-B：4×4 小块内十波段 token 自注意力，只有自由可学习 band ID，无波长条件、无关系偏置。",
        "自由 ID 经同尺寸 width×width 线性层，与波长嵌入层可训练参数一致。",
        "位置：十波段入口前残差修正，修正后同时进入主路与细节分支。",
    ],
    "SegFormer-B0-S3B-IDAttention",
    "spectral_relation",
    REL + ["  relation_mode: none", "  identity: channel"],
)
write(
    "s3_c_wavelength_attn.yaml",
    [
        "S3-C：同一十波段 token 自注意力，改由真实中心波长的 Fourier 嵌入提供 band 身份，无关系偏置。",
        "与 S3-B 只差 identity；两者身份嵌入层可训练参数完全一致。",
    ],
    "SegFormer-B0-S3C-WavelengthAttention",
    "spectral_relation",
    REL + ["  relation_mode: none", "  identity: wavelength"],
)
write(
    "s3_d_local_bias.yaml",
    [
        "S3-D：同一注意力 + 波长身份 + 每个 4×4 小块实测的局部物理关系偏置（差、归一化差、波长距）。",
        "关系偏置末层零初始化，初始注意力等于普通自注意力；读出零初始化 → 初始恒等。",
        "该组同时作为 S4-INPUTRES：融合位置是入口前残差，不重复训练。",
    ],
    "SegFormer-B0-S3D-LocalBias",
    "spectral_relation",
    REL + ["  relation_mode: measured", "  identity: wavelength"],
)
write(
    "s3_e_global_bias.yaml",
    [
        "S3-E：同一注意力与波长身份，关系偏置改为整幅图共享（用平均波段响应），去掉 4×4 位置条件。",
        "与 S3-D 只差偏置是局部还是全局。",
    ],
    "SegFormer-B0-S3E-GlobalBias",
    "spectral_relation",
    REL + ["  relation_mode: global", "  identity: wavelength"],
)

GATE = [
    "spectral_gate:",
    "  patch: 4",
    "  width: 64",
    "  heads: 4",
    "  layers: 2",
    "  relation_mode: measured",
    "  identity: wavelength",
]
write(
    "s4_add.yaml",
    [
        "S4-ADD：真正的十波段关系表示 F_spec 在官方解码特征处同尺度相加，无门控。",
        "视觉主路仍读原始十波段；alpha 与 F_spec 投影零初始化，初始零贡献。",
        "预算：停止 2000 次成功更新，学习率日程沿用旧 3600 步（预热 180）。",
    ],
    "SegFormer-B0-S4Add",
    "spectral_gate",
    GATE + ["  fusion: add"],
)
write(
    "s4_fixed_gate.yaml",
    [
        "S4-FIXEDGATE：同一 F_spec 与融合位置，门控固定为 0.5，不学习。",
        "固定门没有可训练门参数，不能把它与学习门当成同有效容量。",
    ],
    "SegFormer-B0-S4FixedGate",
    "spectral_gate",
    GATE + ["  fusion: fixed", "  fixed_gate: 0.5"],
)
write(
    "s4_learn_gate.yaml",
    [
        "S4-LEARNGATE：同一 F_spec 与融合位置，门控为输入依赖的空间 3×3 卷积（零初始化 → 初始 0.5）。",
        "记录 gate 均值/方差、alpha、修正幅度、目标/背景与 FP 上的分布。",
    ],
    "SegFormer-B0-S4LearnGate",
    "spectral_gate",
    GATE + ["  fusion: learn"],
)
write(
    "s4_capctrl.yaml",
    [
        "S4-CAPCTRL：同参数量级的非物理分支控制。",
        "仍用同一 token 注意力与学习门，但 band 身份改为自由可学习嵌入、关闭物理关系偏置；",
        "因此回答“物理身份+关系偏置是否比同等容量的普通关系聚合更有用”。",
        "有效可训练参数与 S4-LEARNGATE 相差约 0.1%（关系偏置 MLP 的 132 个参数）。",
    ],
    "SegFormer-B0-S4CapCtrl",
    "spectral_gate",
    [
        "spectral_gate:",
        "  patch: 4",
        "  width: 64",
        "  heads: 4",
        "  layers: 2",
        "  relation_mode: none",
        "  identity: channel",
        "  fusion: learn",
    ],
)
