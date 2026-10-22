# M1 状态报告：完整双骨干 + 语义 + 粗细分割

> 更新时间：2026-10-17 晚（原首版报告为同日早些时候，当时数据阻塞未解除）
> **结论：M1 = DONE（工程链路 + 真实数据小批量），但性能未评估（属 M2）。**

关键变化：找到了**合法可用的真实数据**——Sen12Landslides（含 B8A）+ 同一 L2A 产品回补
B01/B09。整个数据替换过程见 `docs/data_source_review.md`，
没有放宽任何波段校验，也没有引入任何合成波段。

---

## 1. 逐项验收（对应 §8 清单）

| # | 验收项 | 状态 | 证据 |
|---|---|---|---|
| 1 | 锁定完整 12 波段 S2L2A 可追溯来源 | **DONE** | Sen12 提供 B02–B12（含 B8A）+ 日期 + CRS；B01/B09 由 STAC 同一天同产品回补，item id 记录在 `outputs/logs/real_inference.json` |
| 2 | TerraMind small_tim 真实权重、12 波段、TiM 空间特征、TiM 确实执行 | **DONE** | `smoke_terramind.json`：196 token、(B,196,384)、TiM ON 0.43s / OFF 0.01s、相对差异 78.75% |
| 3 | AnySat base 真实 10 波段 + 日期 → dense，至少两种尺度 | **DONE（真实数据）** | `outputs/logs/audit_sen12.json`：真实 Sentinel-2 上 40m/160m，输出 (B,1536,128,128)，finite=True |
| 4 | 支路在相同地面位置融合 + 像素一致性 | **DONE** | 脉冲一致性测试；另有真实数据交叉验证：STAC 与 Sen12 的 7 个波段相关系数 0.972–0.996 |
| 5 | RemoteCLIP 真实权重、类别原型参与粗定位、语义对齐链路 | **DONE** | `region_text` 损失在真实批次上 `applied=True`；语义相似度按可学习标量注入粗 logits |
| 6 | 粗/细两个可训练头，推理不用 GT 提示 | **DONE** | `predict()` 只接收影像；细化器测试证明其依赖预测粗概率 |
| 7 | forward/backward/save/reload，列出冻结与可训练参数 | **DONE** | 526.0M / 可训练 3.5M；重载 max\|Δ\|=0 |
| 8 | **真实**影像+Mask 小批次完成训练与推理 | **DONE** | `train_full.py --source sen12`：5 步真实训练，loss 1.4510→1.3531，保存 `outputs/checkpoints/last.pt`；`real_inference.py` 5 个样本完整推理 |
| 9 | model_shapes / 日志 / 可视化 / 复现命令 | **DONE** | `docs/model_shapes.md`、`outputs/logs/*.json`、`outputs/visualizations/real_*.png` |
| 10 | 通过后才做减法消融 | 未开始 | 属 M3 |

---

## 2. 真实数据从哪来

| 环节 | 做法 | 证据 |
|---|---|---|
| 掩膜 + 10 波段 | Sen12Landslides `data_harmonized/s2`，每个 .nc 含 15 时相 | 实测 10 波段含 B8A，int16 DN 0–10000 |
| 单期选择 | 用 `pre_post` 属性取灾后时相 | `{pre:8, post:9}`，event_date=2019-03-15 |
| B01/B09 | AWS Earth Search STAC，**只查同一天**，命中同一 L2A 产品 | `S2A_36KVC_20190320_0_L2A`，60m→10m 双线性 |
| 有效像素 | SCL 排除 0/1/2/3/8/9/10/11 | 实测 valid 99.6%–100% |
| 空间一致性 | STAC 与 Sen12 同波段相关（发现 Sen12 数组是转置存储） | B02 0.989 / B04 0.996 / B11 0.985 / B8A 0.972 |

**同一天没有产品就跳过，绝不跨日替代**（本次命中的样本日期完全一致）。

---

## 3. 实际跑通的命令

```bash
# 环境
python scripts/audit_environment.py

# 数据源评审
python scripts/audit_data_contract.py --probe 16      # L4S：确认缺 B8A，严格模式拒绝
python scripts/audit_sen12.py --limit 3               # Sen12：真实 10 波段 + AnySat 前向

# 真实数据完整链路
python scripts/real_inference.py --limit 6            # 真实 12 波段 → 全模型 → 可视化
python scripts/train_full.py --source sen12 --steps 5 # 真实反向 + 保存

# 合成数据接口回归
python scripts/smoke_terramind.py
python scripts/smoke_anysat.py --scales 40 80 160
python scripts/smoke_full_model.py --steps 3

# 测试
python -m pytest tests/ -q                            # 10 passed
```

真实训练日志：

```
[sen12] 发现 503 个 *_s2_*.nc
[sen12] 跳过 4 个样本（post 时相无滑坡像素）
[sen12] 构造 6 个真实批次（12 波段含 B01/B09 回补）
[step 1/5] loss=1.4510 fine=0.8139 coarse=0.8723 lat=0.73s
[step 2/5] loss=1.4201 fine=0.8077 coarse=0.8630 lat=0.14s
[step 3/5] loss=1.3733 fine=0.7946 coarse=0.8497 lat=0.13s
[step 4/5] loss=1.3569 fine=0.7879 coarse=0.8421 lat=0.13s
[step 5/5] loss=1.3531 fine=0.7876 coarse=0.8394 lat=0.13s
[ckpt] outputs/checkpoints/last.pt
```

真实推理（未训练权重）：

```
[real] chimanimani_s2_1001.nc gt=27  pred=16384 align_corr=0.974
[real] chimanimani_s2_1003.nc gt=242 pred=16384 align_corr=0.985
[real] chimanimani_s2_1005.nc gt=310 pred=16384 align_corr=0.965
```

**这些 IoU≈0 不是模型能力的结论**：只有 5 步、5 个样本、权重基本还是初始化状态。
M1 只要求链路可运行；精度属 M2。

---

## 4. 仍然存在的问题（不阻塞，但会影响 M2）

1. **样本量小**：本地只有 `s2_part01`（503 个 .nc），全量需 28 个分卷 ≈ 39 GB。
2. **小目标**：单样本正像素 27–362，滑坡占比 <1%，对损失与采样策略敏感。
3. **云**：命中的产品云量 49%，SCL 会剔掉大片区域，有效训练面积有限。
4. **B01/B09 是插值**：60 m→10 m 不增加细节，TiM 的高频输入仍是插值产物。
5. **Sen12 数组转置**：内部自洽，但任何外部数据回补都必须转置，已在代码中固化并写下依据。
6. **区域划分**：目前只用了 `chimanimani` 一个事件，跨事件泛化未验证。

---

## 5. 下一步（M2）

1. 下载全量 s2 分卷（或先多取 5–10 个 part），按 region 做事件互斥划分；
2. 冻结骨干、训练融合/语义/两个分割头，记录 mIoU/Dice/Recall/FPR；
3. 针对小目标调整采样与损失权重；
4. M2 稳定后再做 M3 减法消融（−TiM / −AnySat / −SemAlign / −SemLoc / −SemRef / −CoarseFine …）。
