# 02\_EncoderV3\_LatentResponse\_Agent\_Execution\_Plan (1)

# EncoderV3-LR v1.2：Agent 分阶段重构与训练前验收计划

> **任务定位**：按已确认规范重构 EncoderV3；未定接口先提案、确认后实现，但**仅授权设计、编码、自动化测试、真实数据无更新 forward/backward 与报告**。本文件**不授权**任何真实数据 `optimizer.step()`、长周期训练、历史检查点删除、复用既已停止的预检预算。通过第一层工程验收后，Agent 应停在授权边界报告结果。
>
> **必须联合阅读**：同目录 [01_EncoderV3_LatentResponse_Architecture.md](01_EncoderV3_LatentResponse_Architecture.md)，其中的 v1.2 已确认架构合同与 §14 待审接口共同约束实施。旧版本“逐通道独立 P32 / 32 个维度 Token 自选择 / 支持影像推断器 / 共享原型码本”提案**全部作废**。
>
> **代码基线**：`niupai1001/disaster-detect`，参考分支 `dev/optical-intra-selfsup-v1`，审阅提交 `7d1db632c40a8dd7c6264cb95d97ec03235b2b61`，源码根 `earth-representation/encoder_v3/`。实际执行前检查 HEAD 与工作区状态；若已变更，先记录影响文件和差异，不覆盖新进展。
>
> **唯一架构身份**：`latent_response_sensor_memory_v1`（新身份，不与旧 `latent_response_v1` 或 `participation_selected_windows_v1` 的 checkpoint 混用）。

> **本次交付状态**：仅修订文档，没有执行仓库修改或模型测试。原 SHA 为已有审阅基线，不代表重新核实后的当前 HEAD。原始上传的 v1.1 文件保留。
>
> **规格身份**：`spec_revision=1.2`；将两份核心规范 hash 与已确认接口记录绑定运行身份。`architecture_id` 保持架构族名称不变，不构成与其他规格/检查点数值兼容的承诺。

## A0. 本轮变更与首个交付边界

**已确认并须同步落实**：取消首版独立逐像元/逐通道高维 skip，包括旧文中的 micro-cell fine skip；以融合 summary 为主要表示。A 完全隐藏区域不在 Encoder 引入可学习空白内容 Token，Decoder 从可见 summary 和全局 context 读出，并显式接收目标相对位置、支持尺寸与目标通道条件。原有融合后 `R(C)` 局部 latent 保留，不能把取消 skip 扩大成删除它们。

**第一阶段可以立即进行**：核查实际 HEAD/工作区、数据身份和目录证据，完成不依赖待审选择的 Registry、MemoryProjector、动态卷积、纯几何基础和独立测试；对未给定的 Query 编码小投影等细节也必须先列入接口提案，不任意补全。

**第一阶段必须提交**：`interface_proposal.md`，集中覆盖架构 §14 的 I1–I5：标准化与损失、LocalEmbed/未定小投影、空间位置/聚合/回注/流式、Decoder 读出细节、目录和完整场景口径。每项写清候选、推荐理由、张量/掩码、资源影响和验证方法。依赖未确认选择的正式实现暂缓；不依赖部分继续。尚未确认时，应报告对应 `BLOCKED` 项，不把整条 E1 记为 PASS。

**确认后再完成**：完整 D4–D8 集成及真实数据 E1 无更新验证。每项接口的确认范围单独记录；本轮认可不等于批准标准化候选公式、独立 skip 消融、真实更新或正式训练。未来 skip 消融另立协议和授权，不在首版预埋启用路径。

## A. 不允许 Agent 自行改变的架构合同

### A1. 用户已明确决定的部分

1. **每个卫星载荷仅一个持久化 P32**：`SensorKey=(platform,instrument)` → `nn.Parameter([32])`；同一载荷在全部场景、波段子集、A/B1 视图中复用它，训练中仅初始化一次。**不创建每通道/每图像描述器，不创建随高光谱波段数增长的 Embedding 词表。**
2. **P32 是 KV 的来源，而不是各模块 Query**：共享 `MemoryProjector(P_s)` 生成多个 KV 槽位；空间卷积、通道融合、目标 Decoder 由各自**全载荷共享**的模块 Query 读取。同一个 P32 的 32 维不按 PSF/SRF 人为分组。
3. **模块 Query 区分通道**：`Q_t(s,c)=q_t^0+f_t(meta(s,c))`，其中 `meta` 包含在载荷固定目录中的 canonical 顺位及原生 `log(GSDx/GSDy)`；即使 S2 同时输入 B02/B04/B08，仍只能使用一份 P32，但 Query 必须可区分。**同一通道在十波段输入和四波段子集中的固定顺位不改变。**
4. **空间动态卷积**：模块 Q 读取载荷 KV 后产生条件，组合全载荷共享的卷积基，逐通道原生 3×3 输出 32 维；无载荷专用卷积核。
5. **通道融合**：同一个 KV 被通道融合 Query 读取；条件注入 Sub-Patch Channel Cross-Attention；得到 1 个长距离空间 summary 和随有效通道数变化的 `R(C)` 个**局部潜在通道 Token**。
6. **自监督目标也要区分通道**：Decoder 不是简单接收 `P_s`，而必须用**目标通道**的 `Q_decoder` 读取同一 KV，生成不同目标条件；B1 隐藏值不能经 Query、归一化统计或融合 K/V 泄漏。
7. **一个共享 Encoder**，不按 S1/S2/Landsat/EnMAP/GF 划分模型或网络参数；不要求 SRF/PSF/MTF 作为本轮模型前向输入，也不运行“有物理信息/没物理信息”双分支。物理教师暂不实现。
8. 保留原生地理网格和真实支持，通过融合表示学习细尺度信息，不将卷积后的 32 维中间态等同于原生细节，也不承诺无损恢复。首版不保留独立逐像元/逐通道高维 skip，不重采样原始粗 GSD 波段；大场景采用整景语义等价的流式/分块执行，**不通过固定裁剪或旧 16,384 原生像元上限过滤 S2 十波段来规避计算问题**。
9. 首版训练任务仅 A+B1，保留源/观测/目标通道层次损失；禁止 Agent 自行加入 B2、JEPA、SAR 真实训练、测试时支持集编码器或替代损失。
10. 新模型、配置、实验检查点与训练准入身份单独保存；不把旧报告/旧门禁视为已经授权新版本训练。
11. A 完全隐藏区域使用 Decoder 上下文读出：可见 summary + 全局 context + 目标相对位置/支持尺寸 + 目标通道条件；不在 Encoder 新增可学习空白内容 Token。可见输入支持、目标有效性、上下文可用性分别记录。

### A2. 为可执行性冻结的 v0 工程默认值（不是验证后的最优值）

| 对象 | 固定实施值 | 备注 |
| --- | --- | --- |
| 传感器描述 `P_s` | `[32]`，`Normal(0,0.02²)`，按稳定 SensorKey 初始化一次 | 用户已决定 P32；初始化是默认 |
| 注册粒度 | `platform + instrument`（S2A/MSI、S2B/MSI、L8/OLI 等） | **不是** `platform+band_id` |
| MemoryProjector | 共享 `Linear(32,64) → GELU → Linear(64,512)` | 512=`2×8×32` |
| KV 数 | 每载荷 `K,V∈R^[8,32]` | 同一个 P 派生；不是 8 个独立描述器 |
| Query | `q_conv/q_channel/q_decoder` 三个共享种子 + 对规范化顺位 Fourier 编码、`log(gx),log(gy)` 的各自小投影 | **不用当前张量 index**、不读取目标像元 |
| Conv 核基 | `W0` + `8` 个共享 `[32,1,3,3]` 基核；`softmax(Linear(z_conv))` 组合 | 只有一次从 P/KV 的 QKV 读取；避免多余注意力层 |
| Native feature | 3×3 stride 1、mask-aware、GELU，输出 32 | 前端中间态，不是独立 Decoder skip |
| 首版 skip | 无独立逐像元/逐通道高维旁路；无 micro-cell fine skip | 融合后的局部 latent 仍保留；skip 消融另立协议 |
| A 空白区域 | Decoder 查询可见 summary + 全局 context；目标相对位置和支持尺寸进入预测 | 无 Encoder 空白内容 Token；详细读出算子见待审 I4 |
| Sub-Patch | `g_ref=min_{full_geometry}(gx,gy)`，`L_sub=8g_ref`；每 Sub-Patch `4×4` micro-cell；米制、north-up | B1 隐藏通道不改变全场几何参考网格 |
| 融合与主干宽度 | `D=128` | 可部分复用旧配置理念，不载入不匹配旧权重 |
| Local spectral Token 预算 | `R(C)=min(16,ceil(sqrt(C)))`，`C=0` 仅非内容几何占位 | C=1/2/4/10/224 → R=1/2/2/4/15；A 目标不因此取消 |
| 空间骨干 | 8×8 local window、细层 2×4 heads；4×4 粗聚合；32 个全局 latent、2×4 heads；1 层局部谱交互和全局回注 | 不运行整景完全全局 attention |
| 初始优化建议 | FP32 / AdamW；旧 E0 peak LR 3e-4、weight decay 0.04、clip norm 3 只作候选 | Agent 不得无依据自行改变数值，后续训练配置另有授权 |
| 首版真实数据 | 已审 TRAIN/VAL 的 S2/Landsat；仅合成数据检查 224 通道和 SAR 类型兼容接口 | 新数据接入另立工作 |

如果发现固定默认值实测不可运行：**留下可复现的阻塞证据并报告最小变更建议，不得隐蔽变更网络深度、R(C)、通道定义、训练集或任务协议。**

## B. 现有代码复用/替换清单（已按仓库核查）

| 代码路径 | 真实现状 | 新方案如何处理 |
| --- | --- | --- |
| `model.py:BandObservation` | 2D 原生 reflectance、valid/visible、SRF 数组、仿射与 CRS；仅当前米制 north-up | **保留旧类型**；增加无需 SRF 的新 ChannelObservation 适配层，不改旧用法 |
| `data.py:NativeOpticalDataset` | TRAIN/VAL、缓存 hash、原生网格、真实 SRF 文件源身份校验 | **复用 S2/Landsat 真实数据加载**；数据层仍允许历史 SRF 审计，但 SRF 不传给新 Encoder 条件 |
| `tasks.py` | `spatial_task`、`spectral_task`、固定 B1、确定性种子 | **复用任务选择语义**；新 task adapter 保证先隐藏，再任何可学习运算 |
| `formal_schedule.py` | 原型 E0 冻结采样和遮盖；旧 `max_native_pixels=16384` | 保留确定性计划算法；新 sampling identity/pool，**不沿用旧 pixel-cap 过滤** |
| `model.py:SRFEncoder`、旧 `hyper` | 完整 SRF 32D + log-GSD → 3×3 核 | **新 SensorRegistry + MemoryProjector + Query Reader + kernel bank 替换**；旧模块仍在旧类中 |
| `participation_encoder.py`, `participation_layout.py` | per-band Participation Merge、局部网络和 complete global attention | **不继承、不复用 Token 主干或 Merge 曲线**；只借鉴 grouped conv、原生支持测试与执行优化 |
| `layout.py`, `spatial_index.py`, `geometry_cache.py`, `participation_geometry.py` | 原生支持、精确几何/面积与空间索引 | **抽取适用的纯几何函数**，建立 Sub-Patch 新缓存身份；不能将 Merge group 当新 Token 支持 |
| `model.py:ReflectanceDecoder`, `participation_decoder.py` | 精确原生 readout、support mask、cell-indexed 加速；目标条件是 SRF32 | **复用几何/分块读出原理**；新 decoder 必须使用目标 Q\_decoder→KV 的条件（而非共用 P\_s 直接查询） |
| `streaming.py`, `adaptive_streaming.py`, `joint.py`, `losses.py` | exact chunked supervised losses、A/B1 累积、分层汇总、部分假设旧 Representation | 挑选与模型无关的分块/损失逻辑复用；重建必要适配；不能继承旧 Merge 计划 |
| `validation_metric.py`, `validation_panels.py` | 固定面板与 A/B1 层次标量 | 统计口径复用，**新建新架构绑定的 panel**，旧三观测面板不作为充分覆盖 |
| `training.py`, `route_control.py`, `route_step.py`, `formal_runner.py` | AdamW、事务检查点、逻辑更新、恢复、early-stop；存在验证后状态持久化疑点 | **复用控制逻辑、单独修复状态事务**；模型/P Registry/目录/采样新身份必须纳入 checkpoint |
| `gates.py`, `selfsup_preflight.py`, `selfsup_train.py` | 旧 E0 fail-closed 八项门禁/预检和受控入口 | 旧代码与门禁不改，**为新架构建立更轻量但严格的新准入/入口** |
| `README.md`, `AGENTS.md`, `reports/optical_selfsup_closure/` | 历史记忆与证据 | 仅新增新架构索引或附录；旧实验结论不可重标为新结果 |

**既存风险**：`formal_runner.py` 在 `execute_route_update(... runner_state=dict(state))` 时复制状态，`route_step.py` 之后才执行验证回调，checkpoint 可能保存验证前的 early-stop 值。新 runner 必须在验证完成后写入**最新 live state**并有恢复测试；不能仅凭旧“测试通过”报告接受。

## C. 建议创建的目录与统一 API

所有新实现置于 `earth-representation/encoder_v3/latent_response/`，旧实现不覆盖。

```text
encoder_v3/
  latent_response/
    __init__.py
    config_lr_e0_v1.json
    types.py                 # SensorKey, ChannelMeta, ChannelObservation, LRRepresentation
    registry.py              # SensorKey -> 一组持久化 P32 + ChannelCatalog（无 per-band Parameter）
    sensor_memory.py         # 一组 P32 -> 多槽共享 KV
    module_query.py          # Q_conv / Q_channel / Q_decoder，module+channel 静态元数据
    dynamic_conv.py          # Qconv 读 KV -> 共享 8 核组合 -> masked native conv
    native_patch.py          # 地理 Sub-Patch，4x4 micro-cell，真实面积与流式支持
    channel_fusion.py        # Qchannel 读 KV、构建 Channel K/V、输出 summary + R(C)
    spatial_backbone.py      # local/coarse/global latent / feedback
    model.py                 # LatentResponseEncoder（只有一条模型路径）
    decoder.py               # Qdecoder 读同一 KV，目标区分+原生标量读出
    data_adapter.py          # 旧 optical loader -> 新输入；SensorKey/ChannelCatalog 生成
    task_adapter.py          # A/B1 遮盖及新目标计划，不读隐藏目标值
    schedule.py              # 新采样与整景执行计划
    preflight.py             # check/no-update；永不 optimizer.step
    train.py                 # 独立显式授权的真实训练入口
    tests/
      test_registry_memory.py
      test_module_queries.py
      test_dynamic_conv.py
      test_native_patch.py
      test_fusion_backbone.py
      test_masking_decoder.py
      test_recovery.py
      test_real_no_update.py
    reports/                 # 小型可读诊断和结果，不放原始影像
  docs/
    LR_ARCHITECTURE.md       # 本次修订后的 01 文档
    LR_AGENT_PLAN.md         # 本次修订后的 02 文档
```

```python
@dataclass(frozen=True)
class SensorKey:
    platform: str
    instrument: str

@dataclass(frozen=True)
class ChannelMeta:
    channel_id: str
    canonical_ordinal: int      # 按原生载荷完整通道目录稳定编号
    catalog_total_channels: int # 不随输入子集/遮盖变化

@dataclass
class ChannelObservation:
    sensor_key: SensorKey
    channel_meta: ChannelMeta
    values: Tensor              # [H,W] 标定后经 TRAIN-only 统计标准化的标量
    valid: Tensor               # bool[H,W]
    visible: Tensor             # bool[H,W]
    affine: tuple
    crs: str

class SensorRegistry(nn.Module):
    # register one P32 per SensorKey; no ChannelKey nn.Parameter
    def get(self, sensor_key: SensorKey) -> Tensor: ...

class SensorMemory(nn.Module):
    def forward(self, p32: Tensor) -> tuple[Tensor, Tensor]:
        # K, V: each [8,32]
        ...

class ModuleQuery(nn.Module):
    def forward(self, kind: str, meta: ChannelMeta, gsd_xy: tuple) -> Tensor:
        # kind in {conv, channel, decoder}; shared per-kind query parameters
        ...

class LatentResponseEncoder(nn.Module):
    def forward(self, channels: list[ChannelObservation], *, origin_xy: tuple,
                full_geometry: object, execution: str = 'streamed') -> LRRepresentation: ...

class LatentResponseDecoder(nn.Module):
    def forward(self, representation: LRRepresentation, *, targets: list,
                sensor_key: SensorKey, target_meta: ChannelMeta, target_geometry: object): ...
```

**注意**：v1.1 执行计划将 `full_geometry` 定义为完整输入场景的非数值通道网格元数据；架构原文另有“完整通道目录几何”的表述。实际取值集合由 I5 根据数据合同核实并提案，不默认为二者等同；B1 隐藏通道不得修改经确认的原始参考几何。`ModuleQuery` 的 `kind` 是固定三模块类型，**不是卫星或模态分支**。`execution` 只切换数值等价执行后端，不切换模型拓扑。允许 Agent 在不改变上述语义的前提下按 Python 习惯调整类名，但应在变更报告中列出名称映射。

## D. 按依赖顺序实施（阶段 D0–D8）

### D0 — 分支、身份及文档冻结（无参数更新）

- [ ] 记录实际 git SHA、dirty files 与环境依赖；基于当前 HEAD 创建不覆盖旧代码的新工作分支（如 `dev/encoder-v3-sensor-memory-v1`）。
- [ ] 将这两份 v1.2 MD 复制到 `encoder_v3/docs/`，更新项目 `AGENTS.md` 索引，**只引用新架构身份与本规格版本**；旧模型和历史报告保持可读。
- [ ] 先提交 `interface_proposal.md`（I1–I5）及现有元数据证据；确认记录未齐全时不默认补全依赖接口。记录规格 hash 和确认范围。
- [ ] 跑旧现有 CPU 小型回归或 `python -m encoder_v3.verify`（按可运行环境），记录 baseline；不重复昂贵历史 GPU 检测。

- **完成证据**：新 branch、base SHA、现有更改清单、新架构/配置身份；旧模型可独立导入。

### D1 — 一载荷一 P32 的 Registry + 固定 ChannelCatalog

- [ ] 从现有 `row.sensor`、`row.instrument` 等元数据建立 `SensorKey=(platform,instrument)`；S2A/B 若数据元信息确实区分则分别注册，不按短写 `S2` 混合未知 identity。若无法唯一确定载荷身份，标记数据合同阻塞，不能由 Agent 猜测。
- [ ] 为每个 SensorKey 建立**一组**独立 `nn.Parameter(32)`（稳定 key 排序与 seeded 初始化），不为每个 `channel_id`/影像生成参数。目录范围未确认时可用明确标为合成的 fixture 测试，不把假设目录当作真实数据已准入。
- [ ] 实现每载荷的不可学习 `ChannelCatalog`：包含所有合法通道名称、在原生产品内的稳定 `canonical_ordinal`、`total_channels`；即使只载入四个波段，也使用经 I5 确认的完整目录位置；不得把当前四/十波段数组长度直接当作载荷全目录大小。目录只作结构元数据和 Hash，不是 Embedding 词表。
- [ ] 实现保存/恢复：P 参数、目录与注册排序、optimizer state 和 RNG；未知载荷/未知通道 fail-closed，不默默使用默认 P。新载荷注册在新任务/新 run 前完成，运行中不热加入优化器。
- [ ] 当前 batch 未使用的载荷 P 保持 `grad is None`，并在 AdamW 下验证**未访问载荷参数不发生变化**（包括动量与衰减）；使用 `zero_grad(set_to_none=True)`，避免密集 Embedding 全表更新。

- **完成测试**：S2 十/四波段都引用同一 P；L8 引用另一 P；P 总数==载荷数；224 通道合成场景**不会产生 224 个 P**；通道子集、置换和 checkpoint 重建键一致。

### D2 — 从 P 生成多槽 KV；模块和通道 Query 读取

- [ ] `SensorMemory`: 共享 `32→64→512`，输出 `K/V` 各 `[8,32]`；K/V 为**唯一 P32 的派生张量**，不是独立 `nn.Parameter` 表，不复制为 per-band 描述器。
- [ ] `ModuleQuery`: 定义三种全载荷共享的 `q_conv^0`、`q_channel^0`、`q_decoder^0` 及对应小投影；输入规范化稳定顺位的 Fourier 特征、`log(gx)`, `log(gy)`；不从当前通道的真实像元生成 Query，不使用传感器 One-Hot、波长占位符或未知 PSF。
- [ ] 实现一次 Q/K/V 读取 `z=softmax(Q K^T/sqrt(32)) V`，每个载荷的 KV 只计算一次并可供本次 view 内多个通道/模块复用，缓存须保持计算图完整且不跨 optimizer step 使用陈旧张量。
- [ ] 验证不同 `Q` 能对 8 个槽位形成不同权重；**不得退化为 K/V 长度为 1**。注意力对 P32 全维传梯度，不硬划分维度或增设任务专用 P。

- **完成测试**：同一 S2-B04 在四/十波段输入的 Query 完全一致；B04/B08 的 Query 可区分；Q\_conv/Q\_channel/Q\_decoder 的权重与梯度确实非平凡；KV 槽位数量为 8；随机初始化固定 seed 可复现。

### D3 — 共享动态卷积与原生 32D 特征

- [ ] 使用 `z_conv` 形成 `alpha=softmax(Linear32→8(z_conv))`，对共享基 `W0 + Σ alpha_m B_m` 组合得 `[32,1,3,3]`；首版**不**在卷积核基上叠加第二个大 QKV 子网络。
- [ ] 逐原生通道执行 3×3 masked conv → GELU → `[H_c,W_c,32]`，保持 `valid/visible` 支持计数、无效位置不读值、Halo=1。
- [ ] 参考 serial 完成后，可实现 grouped/tiled 语义等价优化；大图 GPU 中间激活要能分块重计算/及时聚合。取消 skip 不等于激活已释放；保留完整梯度，不用无依据 detach 换显存。

- **完成测试**：至少两种稳定通道 Query（同一载荷 P32）产生不同核系数的能力；P、MemoryProjector、Query、卷积核基均获得有效梯度；隐藏值注入 NaN/大值不改变可见结果；serial/grouped/tile 在容差内匹配。

### D4 — 原生 Sub-Patch 与真实地理支持

- [ ] 建立非数值 `full_geometry`，即使 B1 把部分通道隐藏，`g_ref` 与固定地理 Sub-Patch 锚点也不能漂移；`L_sub=8*g_ref`、4×4 micro-cell。
- [ ] 原生特征与 micro-cell 按真实交叠面积汇聚，记录 `support_fraction/visible_count`，无支持位置显式 mask；`LocalEmbed` 从 16 个 micro-cell 输出 per-channel `h[j,c,128]`，具体位置敏感运算先按 I2 确认。**不保留 micro-cell skip**，F/U/h 只作为前端中间态，不成为独立 Decoder 输入。
- [ ] 处理不等 GSD、非方形 GSD、非整除边缘、不同原生网格起点、QA holes；当前原生 loader 支持 north-up 米制，其他情形拒绝而非暗中重采样。
- [ ] 流式块边界保持同一地理锚点；**所有符合数据合同的完整场景都可成为训练候选**，不从根本上删去全十波段大图。先按 I5 界定“场景”是完整入库 observation/cache 还是原始产品整幅，不用前者覆盖证明冒充后者。

- **完成测试**：人工可验证的面积权重、全景支持覆盖、同样场景不同分块数值一致；B1 隐藏集合改变时空间 token 网格位置完全不变。

### D5 — Channel Cross-Attention + 分层空间 Transformer

- [ ] `Q_channel` 读取同一载荷 KV 得 `z_channel`，与 `h[j,c,128]`/support 构建真实影像通道注意力 K/V；不能把 `Q_channel` 和 `Q_sum/Q_latent` 混成一种查询。
- [ ] 每 Sub-Patch 输出 1 个 summary 和 `R(C)=min(16,ceil(sqrt(C)))` 个局部潜在 Token；`C=0` 仅非内容几何占位，不在空 K 上 Softmax、不成为可见 K/V、不新增可学习空白内容 Token；`C=1/2/4/10/224` 测得 `R=1/2/2/4/15`。
- [ ] summary 经 8×8 local 2 层、4×4 coarse 汇聚、32 global latents 及 2 层 latent Self-Attention，回注细尺度；局部谱 Token 只与所属/邻近空间 Token 做 1 层交互，绝不展成全景完全 attention。
- [ ] 实现单场景与多场景独立 padding/mask；GPU 大场景前向支持 streaming；不得建立 `N_sub×N_sub` 全景 Attention 矩阵。位置表达、粗聚合、回注和跨块归一化按 I3 已确认算子实现；不能将逐块独立 Softmax 后平均冒充完整注意力等价执行。

- **完成测试**：通道顺序置换不变、不同通道数与缺失 mask、不同 batch 的数值/梯度一致、全局远距离变化能传播、边界没有 Token 丢失；显式记录各级 Token 数。

### D6 — A/B1、目标 Decoder、新数据/验证配方

- [ ] 复用旧 `tasks.py` 生成相同语义的 A/B1：先隐藏值再进入动态卷积；B1 的目标通道不可进入 Encoder、LocalEmbed、Channel K/V。
- [ ] Decoder 按 `SensorKey` 读取**同一份 P32/KV**，用**目标通道固定顺位/目标 GSD**产生 `Q_decoder`，得到 `z_decoder`；它是重建目标的实际条件，**不允许仅以 P\_****s 区分不同目标波段**。
- [ ] Decoder 以融合后 summary 为主要表示，结合可用的融合后局部 latent、全局 context 与目标 `z_decoder32`；无独立高维 skip。目标相对位置、支持尺寸须实际进入读出/预测网络，不止参与索引。具体检索/聚合/坐标编码先按 I4 确认；`Q_decoder` 本身仍仅使用静态通道目录/GSD，不改为逐像元内容 Query。
- [ ] A 完全隐藏 Sub-Patch 无 Encoder 内容 Token，合法原生目标仍从可见 summary 和全局 context 获得预测；不能因直接面积重叠为零而移出损失。分别记录输入支持、目标有效性与上下文可用性；整观测无可见上下文单列报告，不静默换样本或记零损失。旧支持索引只复用适用数学/工程，不继承旧布局假设。
- [ ] 固定 loss `0.5*A + 0.5*B1`，任务内部按目标通道/观测/来源层次平均；禁止调用旧随步数变化的 `curriculum`。只有 TRAIN-only 标准化、现有可追溯数据/QA，不能用 VAL 拟合统计。按 I1 已确认的输入域/输出域/监督域/归约实现；讨论中的“标准化域 MSE”仍是候选，未经确认不得冻结。禁止重复按原始标准差归一化。
- [ ] 训练 pool **取消旧 **`max_native_pixels=16384`** 的资格过滤**；S2 完整十波段成员保留，采用流式全景执行、确定性含波段数和来源覆盖报告，不能静默替代为固定 crop。重新生成新模型的独立 recipe/hash。
- [ ] VAL 新面板至少 4 个不同观测（S2、Landsat 各≥2，S2 至少一个十波段）；每观测有可用 A/B1，固定遮盖与监督支持，保持 TRAIN/VAL 完整隔离。不要求这 4 个场景证明科学泛化，只满足最小工程验证。

- **完成测试**：目标 B04/B08 在同一 P32 下有不同 Query/条件的能力；在保持冻结 TRAIN-only 统计、监督真值、几何和 valid/visible 不变时，仅投毒不可见输入副本，A/B1 输出与同一损失下的梯度保持等价。避免输入/标签共用存储导致真值随投毒改变。覆盖完整隐藏 Sub-Patch 仍受监督、目标几何实际影响预测、mask/面积/计数与层次 loss 精确。

### D7 — 独立预检、原有 checkpoint 缺陷修复与训练门禁

- [ ] `preflight.py --mode check` 只核查模型/注册/目录/数据身份，不跑真实优化；`--mode no-update` 实际 S2/Landsat A+B1 forward/backward，严格不调用 `optimizer.step()`，记录前后参数 SHA 完全一致。
- [ ] 独立 `train.py` 默认拒绝真实更新，只有**显式新架构训练授权**、新配置/数据/注册/代码身份和测试结果齐全才可运行。旧 8 项 E0 gate 不被自动复用；不伪造历史 approval。
- [ ] 新 runner 应保存模型、P32、MemoryProjector、Query、optimizer/scheduler、CPU/CUDA RNG、sampler cursor、最新验证状态与新实验身份；修复旧 runner 在验证前拷贝 `runner_state` 的问题，单测验证完成 → checkpoint → resume → 下一逻辑更新。
- [ ] CUDA 设备选逻辑可见序号；例如环境只暴露物理 GPU1，进程中可能仍需 `cuda:0`，不得盲用旧 `cuda:1` 默认。
- [ ] 配置 checkpoint 频率和最近/最佳保留策略，避免一次更新一个完整 checkpoint 无限堆积；删除只能针对**当前新运行已确认保存成功**的旧检查点，不能删旧实验/未知文件。

- **完成测试**：无更新参数 SHA 相同；训练门禁拒绝未授权；新注册表恢复值与注册身份相同；验证后状态不滞后；中断/恢复不重复视图；单次输入不更新无关载荷 P。

### D8 — 交付、版本记录与报告

- [ ] 将 `interface_proposal.md`、用户确认的接口记录及两份规格 hash 随报告保留；未批准项不得自填通过。
- [ ] 输出 `architecture_manifest.json`, `sensor_registry_manifest.json`, `channel_catalog.json`, `data_pool_report.json`, `mask_and_batch_evidence.json`, `no_update_report.json`, `token_cost_report.json`, `readiness.json`, `implementation_report.md` 到新实验报告目录；不提交私有数据大张量。
- [ ] README 明确 P32→KV→各模块 Q 的真实计算、完整张量 shape、注册粒度、R(C)、为什么目标 Decoder 需要独立 Q、性能/适用范围及未知载荷限制。
- [ ] 报告列清楚旧文件复用/替换关系、全部测试执行命令、Skipped 原因、真实优化器步数、SHA、所有尚未实现的承诺。

## E. 训练前门槛：尽量少，但必须是真实可核查的

采用**E1（Agent 自主执行的无更新工程验证）→ E2（获授权的有界真实更新）→ E3（用户再次授权正式训练）**。不要求论文级消融、所有多源数据已准备或已取得真实 PSF/SRF。

### E1：无真实更新工程验收——依赖接口确认后 Agent 可自主完成

| 项目 | 最低通过标准 | 明确禁止的替代 |
| --- | --- | --- |
| 描述器规模与复用 | `#P32 == #registered SensorKey`；S2 四/十波段都复用该载荷同一 P32；同一载荷 224 通道也只一个 P32；不同载荷各一个 | per-band P32/动态创建大 Embedding 词表 |
| KV/Q 结构与梯度 | `KV=[8,32]×2`；三类模块 Query 可区分；受控非退化测试中条件确实影响相应计算，P/KV/Q/Conv 基梯度有限且连通 | 单 Key Softmax、固定核；用一次非零梯度宣称学习机制已有效 |
| 未参与 P 的参数状态 | 交替载荷优化的 **CPU 合成单步**（经本地合成测试允许），inactive P 的梯度为空且参数不变 | 一次 dense AdamW 衰减所有载荷 P |
| A/B1 隔离 | 固定真值、统计、几何、valid/visible，仅在独立的隐藏输入副本注入 NaN/大值；输出与同一监督损失的参数梯度按 `atol=1e-5,rtol=1e-4` 一致且有限 | 同时改动标签/统计；只检查 mask 存在 |
| 无独立高维 skip | Representation/Decoder 没有逐像元、逐通道或 micro-cell 高维旁路；前端及融合后局部 latent 的梯度完整 | 把 F/U/h 缓存后改名重新传入 Decoder；删除 R(C) 冒充取消 skip |
| A 空白区域及几何读出 | 完全隐藏 Sub-Patch 无 Encoder 内容 Token，但合法目标可从上下文预测并计入损失；相对位置/支持尺寸实际进入网络，具有区分同通道不同目标位置的能力 | 因无直接重叠而忽略 A 目标；新增空白内容 Token；坐标仅用作索引 |
| 目标身份可辨 | 同一 P32、同一地理支持下，B04/B08 的 `Q_decoder` 和读出条件结构可区分；B1 不读取隐藏目标反射率 | 仅传相同 P32 给 Decoder |
| 通道/批处理稳定 | 通道列表置换、batch vs 单独、streamed vs reference 在相同输入上符合 `atol=1e-5,rtol=1e-4`；同时记录最大差和相对范数 | 只测张量 shape、把旧报告作为新通过证据 |
| 原生几何及 Token | 不同 GSD/偏移/QA holes 的面积与支持正确；R(1/2/4/10/224)=1/2/2/4/15；无全景 `N_sub²` Attention | 强制原始重采样、每波段 Token 全部进全局 Attention |
| 真正的 S2/Landsat 无更新 | S2 四波段、S2 十波段、Landsat 六/七波段各≥1 个真实观测，A+B1 前/反向至少一次，finite loss/grad，所有模型/P SHA 前后严格一致，optimizer.step 调用计数 0 | 把随机初始 loss 当作模型质量证明 |
| 大场景数据覆盖 | 新 TRAIN manifest 保留全部符合基础数据合同的十波段成员；至少 1 个**超过旧 16384 像元上限**的完整十波段 scene 做全景前向；一个可承受的完整 scene 完成无更新反向；报告有效空间覆盖与耗时/显存 | 仅中心 128² 裁剪后宣称整景能力 |
| 状态与报告 | I1–I5 已确认、checkpoint/runner 验证后状态测试通过；记录规格/接口/代码/数据/目录身份、真实完成步数、各级 Token 和资源成本 | 自批未定接口、口头全绿或隐藏 ERROR/SKIP |

**误差规则**：以上阈值只适用于同模型、同输入、同语义计算路径对照，不能作为新旧网络相等要求。必须比较对应地理位置/通道而不是误把不同顺序按索引比较。真实大场景反向如因显存不足未完成，记录清楚阻塞；可在 E1 小/中完整景完成其余工程检查，**不能伪称已具备大型整景正式训练能力**。

### E2：有限真实优化器更新/恢复验收（须用户额外授权）

- Agent 在 E1 形成 PASS/FAIL/BLOCKED 记录后停止，不自行申请 GPU 预算或先跑训练。
- 用户允许后，仅使用**新 LR 实验的独立预算**做至少 2 次真实 A+B1 `optimizer.step()`；两次覆盖 S2、Landsat，检查共享 Encoder/Memory/Query 与各自 P 的梯度和更新；与其他载荷无关的 P 不改变。
- 在经过验证的 checkpoint 边界中断/恢复，确认下一成员/掩码/优化器状态、逻辑 update/early-stop 与已完成验证一致；无需强制 GPU bitwise 相等，但要记录浮点差异、避免重复消耗更新。
- 如有条件且获明确批准，可再用 ≤10–20 次更新做短弱学习性检查；**非 E2 强制要求**，不得把过拟合小场景当泛化证据。
- 不通过则只记录并修复真实错误，不擅自更改模型基本结构/学习任务。

### E3：正式 E0 训练（须用户再次明确授权）

仅要求：E1 完成；E2 最小有界真实更新和恢复验收通过；新 S2/Landsat TRAIN/VAL 配方/身份固定、没有旧像元上限隐藏删样本；训练预算/停止条件/验证频率可执行；明确正式训练数据规模对应的反向资源能力与已验证执行范围，小景反向不能代替大景验收；用户发出本架构正式训练授权。

**不要求**先做到 EnMAP/GF/S1 真实训练、未知载荷自动推断、SRF/PSF 教师、JEPA、B2、论文级对照或下游灾害任务精度。

## F. CLI 合同与预期证据（实现后可用；当前不存在）

```powershell
# 在 earth-representation 工作目录执行。先运行全部 CPU/合成测试。
python -m unittest discover -s encoder_v3/latent_response/tests -p 'test_*.py'

# 检查配置/注册/通道目录/数据 hash，不读真实样本，不发生优化器更新。
python -m encoder_v3.latent_response.preflight --config encoder_v3/latent_response/config_lr_e0_v1.json --mode check --device cpu

# 真实光学 A/B1 前向+反向，必须严格无 optimizer.step。
python -m encoder_v3.latent_response.preflight --config encoder_v3/latent_response/config_lr_e0_v1.json --mode no-update --device cuda:0

# 仅在 E1、E2 完成且正式训练被用户授权时允许。
python -m encoder_v3.latent_response.train --config encoder_v3/latent_response/config_lr_e0_v1.json --mode train --device cuda:0 --directory <approved_new_run_dir>
```

`readiness.json` 最少含以下字段：

```json
{
  "architecture_id": "latent_response_sensor_memory_v1",
  "spec_revision": "1.2",
  "spec_sha256": {"architecture": "actual SHA256", "agent_plan": "actual SHA256"},
  "interface_review": {"I1": "PENDING", "I2": "PENDING", "I3": "PENDING", "I4": "PENDING", "I5": "PENDING"},
  "independent_high_dim_skip": false,
  "encoder_learned_blank_content_tokens": false,
  "base_commit": "actual SHA",
  "config_sha256": "new config SHA256",
  "data_sha256": "TRAIN/VAL release identity SHA256",
  "sensor_registry_sha256": "ordered SensorKey + descriptor schema SHA256",
  "channel_catalog_sha256": "canonical channel catalog SHA256",
  "registered_sensor_count": 0,
  "learnable_descriptor_parameter_count": 0,
  "p_dim": 32,
  "kv_slots": 8,
  "e1_preflight": "PASS | FAIL | BLOCKED",
  "e2_update_recovery": "NOT_AUTHORIZED | PASS | FAIL",
  "formal_training_authorization": false,
  "real_optimizer_steps": 0,
  "notes": "Exact tests, scopes, token counts and resource observations"
}
```

报告 token 复杂度应分别记录**原始通道和像元、动态卷积成本、Sub-Patch 数、局部谱 Token 数、粗尺度/全局 latent 数、每视图注意力规模、完整数据准备/前后向/保存耗时**。旧版报告数值只作为历史动机，不得计算虚构加速倍率。

`interface_review` 的 `PENDING` 是模板初始状态，只有对应确认记录存在才可填写 `CONFIRMED`；不能为通过 E1 自动改写。新增字段用于描述本次规格，不代替原有运行身份或训练授权。

## G. 变更控制：遇到阻塞的唯一处理办法

- 遇到波段 P32 方案残留：必须清理新路径所有相关代码/文档；不允许只把 `ChannelKey` 改名但仍按通道增加 nn.Parameter。
- 如果同一 P32 的不同通道卷积完全同核、B1 decoder 无法区分目标：检查规范化 Query/ChannelCatalog/梯度，**不能**重新引入 per-band 参数偷渡身份。
- 如果 SensorMemory 只有一条 KV：这是数学错误，必须扩成多个槽位；也不能用无内容选择的单 Key attention 冒充有效 Q。
- 新载荷未知、无训练曝光：可以提前注册一组 P32 并在后续明确适配任务中学习；不能把随机 P 当作已知响应，也不通过自动支持集推断器掩盖限制。
- 不同卫星数据统计不同：在数据层执行可追溯定标/归一化，不增加按模态切换的网络模块。
- 全景大图 OOM：报告哪些计算/激活占显存，优化流式、Halo、重计算；**不**固定裁图、排除全十波段、回退旧全局 Transformer 或偷偷降低通道数。
- 取消 skip 不等于取消融合后 R(C)，也不等于禁止现有全局 latent Query；不得擅自添加原始标量旁路或未来消融配置。
- 未定接口按 A0 / 架构 §14 一次性提交具体提案；不能用“工程默认”替代用户确认。
- 任一设计改动影响 P32 粒度、KV 数、Query 来源、动态核公式、R(C)、任务/损失、空间支持、采样资格、训练授权：先单独说明拟改理由与影响，并请求用户确认；不得 Agent 自己改后当作既定方案。

## H. Agent 最终交付报告固定格式

```text
1) 实际 base HEAD、工作分支、保留的既有工作区变动
2) 规格 v1.2 / 文件 hash、I1–I5 的提案与确认状态、D0–D8 的完成/失败/阻塞证据
3) SensorKey → 一个 P32；多个 KV；Q_conv/Q_channel/Q_decoder 的张量和梯度检查
4) S2 四/十波段复用 P 的证据、Landsat 独立 P、inactive P 不被更新
5) 无独立高维 skip、A 完全隐藏区域读出/目标几何、固定真值的投毒、通道/批处理/流式一致性
6) S2 4-band、S2 10-band、Landsat 的真实无更新前后向检查
7) Token 数/整景覆盖/峰值显存/实际时间（不混淆估计与实测）
8) 旧代码保留和实际复用位置、旧 Runner 恢复缺陷修复证据
9) E1/E2/E3 状态、是否发生真实 optimizer.step、是否得到授权
10) 真实不能完成的事项、需要用户作决定的最少问题及复现命令
```

**分阶段交付界线**：第一阶段交付接口提案、数据证据、已明确模块的代码/测试及剩余阻塞，不能因此宣称 E1 完成。待依赖接口获确认后，完整 E1 无更新检查达到 PASS、E2 显示 `NOT_AUTHORIZED`，代表本轮**实现与工程预检**完成；直到用户另外授权才准执行 E2 实际优化器更新或 E3 正式训练。本次仅交付文档修订，没有生成或运行上述模型代码和测试。
