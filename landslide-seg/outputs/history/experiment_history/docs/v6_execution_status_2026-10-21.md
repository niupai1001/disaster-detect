# v6 执行状态与续跑说明（2026-10-21 06:30）

本文是运行状态记录，不是结果报告。结果见 `docs/v6_temporal_spectral_2026-10-21.md`。

## 当前状态

| 项目 | 状态 | 关键结果 / 证据 |
|---|---|---|
| 输入/评测/模型契约验收 | 33/33 通过 | `reports/v6_execution/acceptance.md` |
| DOFA 十波段波长修正 4 组 | 完成 | test IoU 0.027/0.014/0.026/0.049，均值 0.029（低于 RGB 0.074 与旧版 0.045） |
| U-TAE 单期真实日期＋DEM | 完成 75 epoch（best@50） | val 0.3772，test 0.3396，AP 0.5142 |
| U-TAE 15 期真实日期＋DEM | 完成 75 epoch（best@55） | val 0.4873，test 0.4832，AP 0.6853；差值 +0.1436/+0.1711 |
| 条件组：截至单期日期的 U-TAE | **训练中**（val 差 +0.1101 > 0.02 触发） | `outputs/utae_v6/utae_history_realdates/1234` |
| 主模型单期参照 | 完成 3600 更新（best@2300） | val 0.3587，test 0.3302，AP 0.4816 |
| 主模型 15 期无光谱 | 完成 3600 更新（best@2600） | val 0.4149，test 0.4093，AP 0.6032 |
| 主模型单期光谱 | 完成 3600 更新（best@3600=last） | val 0.3705，test 0.3336，AP 0.4811 |
| 主模型 15 期光谱 | **训练中** | `outputs/segformer/v6_main/SegFormer-Temporal-Multi15-Spectral/1234` |

## 本轮修正的两处工程问题

1. **多期光谱组启动即崩（`Memory allocation failure` / `cudaErrorUnknown`）。** 原因是该组配置单独加了 `data.preload: true`，而 `TemporalV6` 的 preload 会把 3012×15×10×128×128 的 float32 张量（约 37 GB）整块读入内存。该字段在基础配置与其余三组都不存在，因此既导致 OOM，也破坏四组「同一配置」的可比性。已删除该覆盖项，按其余三组同样的懒加载重跑；崩溃残留的三个元数据文件归档到 `outputs/segformer/v6_main/_partial_restart/SegFormer-Temporal-Multi15-Spectral_oom/`，未删除。
2. **多期组单图推理延迟记录与物理直觉矛盾。** 记录值 12.26 ms（15 帧）小于单期 16.91 ms（1 帧）。独立复测得 T=1 17.3 ms、T=15 61.5 ms，报告采用复测值，旧记录标记待复核；`scripts/evaluate_temporal.py` 的 `_latency` 本身确实按 `t=15` 构造输入，问题更可能在测量环境（同机并发/时钟状态），尚未定位。

## 环境

2026-10-21 02:48 出现的 `nvlddmkm` Event 153 已随重启恢复；本轮两次启动 CUDA 任务均正常，双卡可用（RTX 5090 ×2）。序号 0 卡在崩溃后曾短暂显示 18 GiB 占用但 `mem_get_info` 报 30.3 GiB 空闲，属缓存态，未影响启动。

## 续跑方式

单条命令串行跑完剩余的评测与汇总（已完成的 train_summary 会跳过）：

```powershell
cd N:\Users\yzh\Disater\landslide-seg
& 'C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe' -X utf8 scripts/run_v6_remaining.py
```

- GPU0 链：主模型 15 期光谱（已完成则跳过）→ best/last 评测；
- GPU1 链：U-TAE「截至单期日期」组 → 评测；
- 末尾 `scripts/summarize_v6.py` 刷新 `reports/v6_execution/summary.json` 与报告用图。

注意：`run_v6_remaining.py` 会对所有已存在 checkpoint 的组重复评测（确定性一致，但会额外占用 GPU 时间）。
