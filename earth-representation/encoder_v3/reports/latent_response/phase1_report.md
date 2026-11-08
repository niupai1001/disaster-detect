# 第一阶段交付报告 — EncoderV3-LR v1.2

日期：2026-11-08。架构身份：`latent_response_sensor_memory_v1`；规格版本
`spec_revision=1.2`。本报告只记录本轮**实际执行**的内容；未执行项单列。

## 1. 代码基线与工作区

| 项目 | 值 |
| --- | --- |
| 仓库 | `N:/Users/yzh/Disater`（`niupai1001/disaster-detect` 工作副本） |
| 参考分支 | `dev/optical-intra-selfsup-v1` |
| 实际 base HEAD | `7d1db632c40a8dd7c6264cb95d97ec03235b2b61`（与历史审阅 SHA 相同） |
| 工作分支 | `dev/encoder-v3-sensor-memory-v1`（从上述 HEAD 新建，未提交 commit） |
| 保留的原有未提交修改 | `earth-representation/scripts/enmap_pairs_valid.py`（75+/47−，与本架构无关，未触碰） |
| 新实现目录 | `earth-representation/encoder_v3/latent_response/` |
| 规格冻结 | `encoder_v3/docs/LR_ARCHITECTURE.md`、`LR_AGENT_PLAN.md`、`LR_HANDOFF.md`，SHA256 见证据 JSON |

规格哈希（实测）：
- architecture `001ea601e50e11fdacca3c9b6fbaac062beea52e4722d08d7b5d33969711e30b`
- agent_plan `782c868b2781bb8a8ff8787fc51cfaf0f72694b02dbd53057bd225c589e825f7`
- handoff `f0cca2f97c2d05036981239fb7d2d335bdf235385de02562adb64048dfcfe9b8`

## 2. 对架构与当前进展的理解

目标路径是：每个 `SensorKey=(platform,instrument)` 一个持久化 P32 → 全载荷共享
`MemoryProjector` 生成 8 槽 K/V → 动态卷积 / 通道融合 / Decoder 三个模块用各自的
静态通道 Query 读取；原生网格上做 mask-aware 3×3 动态卷积；按 `L_sub=8·g_ref` 的
地理 Sub-Patch 与 4×4 micro-cell 汇聚；通道融合后得到 1 个空间 summary 与
`R(C)` 个局部潜在 Token；分层空间主干（8×8 局部窗口、4×4 粗聚合、32 全局 latent、
回注）；Decoder 用 `Q_decoder(s,c)` 的目标通道条件 + 目标相对几何从可见 summary/
全局 context 读出；首版无独立高维 skip；A 的完全隐藏区域由 Decoder 预测。

仓库现状（核查结论）：
- **旧 Participation 路径完整存在**（`model.py`、`participation_*`、`streaming.py`、
  `losses.py`、`validation_metric.py`、`formal_runner.py` 等），本轮未改。
- **新架构代码在本轮之前不存在**：`git log --all -- earth-representation/encoder_v3/latent_response`
  为空；`git grep latent_response_sensor_memory` 无结果。
- 旧报告中的“通过”只覆盖旧路径；不能作为新架构 E1 证据。旧的 A-hidden 一致性结论
  按 01 §12 仍需在新 Representation 上重新核实。

分类：

| 类别 | 内容 |
| --- | --- |
| 已实现并有对应测试（本轮） | Registry（一载荷一 P32、目录、身份/摘要）、SensorMemory（8 槽 K/V）、ModuleQuery（三模块 × 静态目录/GSD）、动态卷积与 mask-aware 原生卷积、Sub-Patch/micro-cell 纯几何 |
| 已有实现但证据不足 | 旧 loader/tasks/loss/streaming/validation（历史测试覆盖，但未接到新架构）；旧 A-hidden 隔离结论（未重新核实）；旧 runner 验证后状态事务（按 02 §B 仍需新 runner 测试） |
| 尚未实现 | ChannelFusion、空间主干、Decoder、data/task adapter、schedule、preflight、train、E1 入口、新 checkpoint 身份 |
| 需要用户确认 | I1–I5（见 `interface_proposal.md` §6 Q1–Q7） |

## 3. 本轮交付文件

```
earth-representation/encoder_v3/docs/
  LR_ARCHITECTURE.md  LR_AGENT_PLAN.md  LR_HANDOFF.md        # v1.2 冻结副本
earth-representation/encoder_v3/latent_response/
  __init__.py            # 架构身份常量 + 已确认模块导出
  types.py               # SensorKey / ChannelMeta / NativeGrid / ChannelObservation
  registry.py            # 一载荷一 P32 + ChannelCatalog + manifest/identity/digest
  sensor_memory.py       # 共享 MemoryProjector：P32 -> K,V [8,32]
  module_query.py        # Q_conv/Q_channel/Q_decoder（Fourier 频带显式传入）
  dynamic_conv.py        # W0 + 8 基核组合；mask-aware 3x3 原生卷积 + halo 分块
  native_patch.py        # g_ref、L_sub=8*g_ref、4x4 micro-cell 面积聚合
  interface_proposal.md  # I1–I5 提案 + 最少待确认问题 + 矛盾清单
  tests/                 # 4 个独立测试文件（37 项）
earth-representation/encoder_v3/reports/latent_response/
  phase1_evidence.json   # 身份/哈希/测试/数据证据（机器可读）
  phase1_report.md       # 本文件
```

未创建 `config_lr_e0_v1.json`：I1–I5 未确认，写配置会把候选值固化成身份并诱导
误用；待接口确认后再生成并绑定 hash。

## 4. 实际执行命令与结果

工作目录 `earth-representation/`，解释器
`N:/Users/yzh/Disater/landslide-seg/.venv-use-independent/Scripts/python.exe`
（Python 3.12.14，torch 2.11.0+cu128；本轮只用 CPU）。

```powershell
# 1) 新架构独立测试
& $py -m unittest discover -s encoder_v3/latent_response/tests -v
# -> Ran 37 tests, OK

# 2) 旧路径 CPU 回归抽查（不是完整 146 项历史审计）
& $py -m unittest encoder_v3.test_tasks encoder_v3.test_model `
        encoder_v3.test_geometry_cache encoder_v3.test_validation_metric encoder_v3.test_streaming
# -> Ran 21 tests, OK

# 3) 证据生成（读取 release 元数据 + 运行上述两个测试组 + 记录哈希）
#    脚本与输出见 phase1_evidence.json；该次运行 new=37 OK, old=21 OK
```

测试覆盖（37 项）：
- `test_registry_memory.py`（11）：5 个载荷 5 个 P32、224 通道仍 1 个、四/十波段共享、
  未知载荷/通道 fail-closed、确定性初始化与 manifest 往返、inactive P 无梯度且
  AdamW 后不变（CPU 合成单步）。
- `test_module_queries.py`（8）：同通道跨输入子集 Query 相同、B04/B08 可区分、
  三模块可区分、GSD 进入 Query、seed/投影梯度、未用模块无梯度、非法输入拒绝。
- `test_dynamic_conv.py`（9）：核形状/系数归一、不同条件不同核、梯度连通、
  tile 等价、逐样本核与逐次调用一致、隐藏 NaN/大值不影响可见输出、隐藏位置梯度 0、
  可见邻域计数归一化的精确值。
- `test_native_patch.py`（9）：10m/20m/30m 面积权重、跨 Sub-Patch、负索引、
  微格边界、隐藏像元不贡献、无可见单元/非有限可见值拒绝。

关键数值证据（来自测试与数据核查）：
- `#P32 == #SensorKey == 5`（fixture），`sum(numel)=160`；224 通道合成目录仍 1 个 P32。
- KV 形状 `[8,32]×2`；同 P32 的两通道条件产生不同核；B04/B08 的 `Q_decoder` 不同。
- `R(C)` 表在提案中固定为 `min(16,ceil(sqrt(C)))`（本轮未实现融合，未运行）。
- 数据：7778 成员；5 个载荷键；S2 10 通道（10m+20m）、LC08 7、LC09 6；
  TRAIN 10 波段 S2 594 个，其中 586 个 native side>128（与 release.json 排除数一致）。

## 5. 接口状态与停止位置

| 接口 | 状态 | 提案位置 |
| --- | --- | --- |
| I1 标准化/损失 | PENDING | proposal §I1，Q1 |
| I2 LocalEmbed/小投影 | PENDING | proposal §I2，Q2 |
| I3 主干/分块 | PENDING | proposal §I3，Q3 |
| I4 Decoder 读出 | PENDING | proposal §I4，Q4 |
| I5 数据身份/范围 | PENDING | proposal §I5，Q5–Q7 |

本轮停在“依赖接口确认的集成边界”：
- 未实现 ChannelFusion、空间主干、Decoder、adapter、preflight、train；
- 未运行任何真实数据 forward/backward；
- 未创建/加载新架构 checkpoint；
- 未修改旧模型、旧配置、旧检查点或历史报告；
- 未运行 `encoder_v3.verify` 全量历史套件（避免与本轮无关的大范围历史审计），
  只做了上述 21 项 CPU 抽查。

## 6. 优化器步数与授权

| 项目 | 数量 | 说明 |
| --- | --- | --- |
| 真实数据 `optimizer.step()` | **0** | 本轮未读真实样本、未做真实前后向 |
| CPU 合成 AdamW 单步 | 5 | 每次完整测试套件执行 1 次（`test_unused_descriptor_...`），固定 2 个合成 P32、小学习率，与真实数据更新分开记录 |
| E2 有界真实更新 | NOT_AUTHORIZED | 未执行 |
| E3 正式训练 | NOT_AUTHORIZED | 未执行 |
| 历史 400 次 pilot 预算 / 剩余 186 次 | 未使用 | 未触碰 |

## 7. 未执行 / 阻塞证据

- **接口未确认**：I1–I5 均为 PENDING（本报告与提案均未自填 CONFIRMED）。因此
  Decoder/融合/主干/整链路 E1 无法开始；这不是环境失败，而是授权边界。
- **真实数据 E1 未执行**：无真实 `forward/backward`、无参数 SHA 前后对比、无
  Token 数/显存/耗时实测。任何“新架构已验收”的说法都不成立。
- **GPU 未使用**：按 `encoder_v3/AGENTS.md` 记录，此前独立 GPU 进程曾以
  `3221225477` 退出；本轮未重启/未探测 GPU，避免把环境问题混入实现结论。
- **数据范围仅核查元数据**：未读取真实影像缓存内容做模型输入；I5 的范围结论来自
  `release.json`/`members.jsonl`/`contracts`/`npz` 键与统计量，属于数据身份核查，
  不是模型验收。
- **旧 `formal_runner` 缺陷未修**：01 §12 指出的“验证前复制 runner_state”问题属于
  新 runner 的 D7 工作，本轮未改旧代码。

## 8. 下一步（待接口确认后）

1. 按 Q1–Q7 的确认结果冻结 I1–I5，写入 `readiness.json` 的 `interface_review`。
2. 实现 ChannelFusion → 空间主干 → Decoder，接 `tasks.py` 的 A/B1 语义与
   `streaming.py` 的 chunked 反传骨架。
3. 做 D6 的隔离测试：完全隐藏 Sub-Patch 仍有有限预测与监督；同通道不同目标几何
   实际进入网络；固定真值/统计/mask 后仅投毒隐藏输入副本；通道置换、batch、
   分块等价。
4. 真实 S2 4/10 波段与 Landsat 6/7 的 E1 no-update（0 步）与 `readiness.json`；
   E1 前不申请 E2/E3 授权。
