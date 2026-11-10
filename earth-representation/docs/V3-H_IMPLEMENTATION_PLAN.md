# Encoder V3-H：工程实施与代码验收计划

状态：原文档阶段已提交为`73f6b80`。用户随后明确要求“继续完成试验进入训练”，本轮授权实现、工程试验及门禁后的新V3-H训练；旧作业不恢复。工程状态与未通过项以§20和实际证据为准。设计依据：[V3-H_DESIGN.md](V3-H_DESIGN.md)。

## 1. 实施目标

本任务根据 `V3-H_DESIGN.md` 实现完整的层次化混合表征架构。

本文件是工程执行依据。

核心要求：

**代码必须实现设计文档规定的信息流和数据语义，不能仅实现一个形状相似、功能近似的网络。**

特别禁止：

- 将固定8×8写成V3-H空间划分规则。
- 将Fine/Middle/Coarse Token数量固定为16/4/1。
- 通过简单Adaptive Pooling伪装成原生面积聚合。
- 将三个尺度拼接后送入全局密集Transformer。
- 将不同尺度依次压缩生成。
- 让目标像元通过任何旁路进入Online Encoder。
- 为三个尺度重复执行三套原生条件卷积。
- 擅自引入外部目标通道输入接口。

原文档阶段只提交两份文档与索引；当前用户已授权推进下列代码阶段、工程检查及训练。新实现使用独立hierarchical目录与新运行身份，不改变旧冻结源码、成员或已停止预算。

不启动正式训练。

## 2. 开工检查

### 2.1 仓库和分支

确认：

- 仓库：`niupai1001/disaster-detect`
- 分支：`dev/encoder-v3-sensor-memory-v1`
- 工作树状态；
- 当前最新提交；
- 任务认领；
- 历史冻结文件。

不得将其他分支或本地未提交代码当成当前正式基线。

### 2.2 必读文件

- 根目录 `AGENTS.md`
- `earth-representation/README.md`
- `earth-representation/docs/AGENT_ROLES.md`
- `earth-representation/docs/DESIGN.md`
- `earth-representation/docs/RESULTS.md`
- `earth-representation/configs/config_shared_mae_v1.json`
- `earth-representation/src/encoder_v3/shared_mae/model.py`
- `earth-representation/src/encoder_v3/shared_mae/geometry.py`
- `earth-representation/src/encoder_v3/shared_mae/metadata.py`
- `earth-representation/src/encoder_v3/shared_mae/data.py`
- `earth-representation/src/encoder_v3/shared_mae/train.py`
- `earth-representation/tests/encoder_v3/shared_mae/test_shared_mae.py`

先建立当前实现与V3-H设计之间的映射，确认哪些模块可以复用，哪些必须新增。

### 2.3 文件保护

新实现位于：

`earth-representation/src/encoder_v3/hierarchical/`

测试位于：

`earth-representation/tests/encoder_v3/hierarchical/`

配置位于：

`earth-representation/configs/`

不得修改或覆盖：

- 历史shared_mae源码；
- latent_response历史源码；
- 原始观测和QA；
- 已冻结数据成员；
- 历史权重及训练产物；
- 历史实验结果。

优先只读复用公共上游模块。必须修改时先列出具体差异、调用方及影响范围，按后续用户任务的授权处理；不能为方便重写历史shared_mae或冻结发布。

## 3. 代码实现的核心数据契约

### 3.1 原生输入对象

每个原生通道必须明确包含：

- values；
- valid mask；
- NativeGrid；
- PhysicalChannel；
- 通道身份；
- 原生尺寸；
- 设备及数值精度。

不能将不同原生分辨率的通道直接堆叠为一个假定相同H×W的张量。

### 3.2 几何对象

建议建立独立的空间几何对象，至少保存：

- metric patch geometry；
- native pixel footprints；
- source pixel indices；
- destination region indices；
- intersection areas；
- valid support；
- spatial adjacency；
- cross-scale parent-child relations。

几何对象和学习特征必须分离。

可以缓存不随训练视图变化的几何关系，但不能跨遮盖视图复用已学习的特征。

### 3.3 多尺度Token对象

每个尺度应使用明确的数据结构：

- features；
- spatial coordinates；
- support ratio；
- valid mask；
- parent indices；
- child indices；
- original geometry identity。

Fine、Middle、Coarse分别保存，不要在数据层提前拼成一个21Token序列。

实际Token数量由当前观测几何计算。

不同观测可以具有不同Token数量。

## 4. Phase 1：动态几何与三尺度Token生成

这是最重要的第一阶段。

### 4.1 先解决几何划分

当前shared_mae包含：

- 显式地理边长L；
- GSD相关几何；
- 原生像元面积交叠；
- AreaCache；
- 固定内部layout。

V3-H应保留显式L和原生面积交叠，重新设计内部三尺度分区。

空间分区必须由实际几何生成。

不能在代码中写死：

```python
fine_tokens = 16
middle_tokens = 4
coarse_tokens = 1
```

也不能直接写死：

```python
layout = 8
```

作为所有观测的内部空间划分。

Coarse可以采用每个地理Patch一个Token，但Fine和Middle的空间分区必须从实际原生空间支持和尺度层级计算得到。

### 4.2 几何计算要求

给定地理Patch边长L、原生网格和实际像元面积：

1. 构造原生像元与地理Patch的精确交叠关系。
2. 获取Patch内部真实空间支持。
3. 根据Fine、Middle、Coarse的尺度规则生成各自分区。
4. 计算原生像元与各尺度分区的面积交叠。
5. 构建每个Token的有效支持率。
6. 构建跨尺度父子关系。
7. 验证非整除边界和不同GSD。

尺度规则应明确表达为模型配置中的物理尺度层级或分区策略。

**空间层级的数量可以固定为三级，实际分区必须由冻结尺度策略和输入几何计算，不能用统一固定网格或Token配额代替。**

不得将原生数据重采样为统一高分辨率图像后再生成Token。

### 4.3 三尺度并行聚合

输入是同一份原生32维条件特征。

三个分支独立聚合：

```text
NativeFeature[H,W,32]
       │
       ├── aggregate_fine()
       │       └── fine_projection()
       │
       ├── aggregate_middle()
       │       └── middle_projection()
       │
       └── aggregate_coarse()
               └── coarse_projection()
```

三个聚合分支必须：

- 使用相同的原生观测来源；
- 使用各自的空间交叠权重；
- 使用独立的有效支持计算；
- 不依赖其他尺度已经生成的特征；
- 不重复执行条件卷积。

初始通道维度：

- Fine：64；
- Middle：96；
- Coarse：128。

### 4.4 本阶段验收

必须证明：

1. Token数量由实际几何及冻结尺度策略计算；覆盖能改变分区数量的L/GSD/场景组合，不要求所有输入改变都导致数量变化。
2. 不同GSD能够产生正确的空间支持。
3. 三个尺度来自同一份原生32维特征。
4. 三个尺度独立聚合。
5. Coarse可以为每个地理Patch生成一个Token。
6. 面积聚合正确。
7. QA无效区域不参与聚合。
8. 非整除边界正确。
9. 父子关系与真实空间包含关系一致。
10. 没有固定8×8或16+4+1的隐藏假设。

## 5. Phase 2：三个独立Transformer

### 5.1 Fine Transformer

初始：

- 64维；
- 2层；
- 4头；
- Neighbor Attention。

实现要求：

1. 根据真实空间邻接关系构造邻近窗口。
2. 允许跨地理窗口交互。
3. 不允许全场景密集Fine Attention。
4. 对无效Token进行正确屏蔽。
5. 不同观测不能互相注意。
6. MAE目标区域不能通过邻窗交互泄漏。

注意：邻近窗口不等于无限扩展注意力。必须有明确的空间邻域索引。

### 5.2 Middle Transformer

初始：

- 96维；
- 2层；
- 4头；
- Regional Attention。

区域范围由实际地理布局计算。

不允许直接假定所有场景都可以采用固定2×2窗口。

### 5.3 Coarse Transformer

初始：

- 128维；
- 2层；
- 4头；
- Global Attention。

必须：

- 按观测分别执行；
- 不跨场景混合；
- 处理不完整场景；
- 进行最大场景显存预检。

### 5.4 本阶段验收

每个分支必须独立测试：

- 前向形状；
- 注意力索引；
- 有效性掩膜；
- 梯度；
- 场景隔离；
- 邻域交互。

不能仅凭最终输出有数值就认为三个Transformer均已正确实现。

## 6. Phase 3：双向跨尺度交互

实现：

Fine ↔ Middle

Middle ↔ Coarse

要求：

1. 根据几何关系生成父子索引。
2. 使用投影处理不同通道维度。
3. 使用稀疏Cross-Attention。
4. 使用残差连接。
5. 使用有效性掩膜。
6. 保留各尺度原有表示。
7. 不将全部Token拼接后做全局注意力。

需要测试：

- 单个Fine对应多个父子关系时的索引；
- 不完整边界；
- 无效父Token；
- 多个观测Batch；
- 正反方向的信息交互；
- 梯度传播。

必须分别证明Fine→Middle、Middle→Fine、Middle→Coarse、Coarse→Middle均有实际计算路径。

## 7. Phase 4：Hierarchical Decoder

### 7.1 解码路径

```text
Coarse Encoded Features
          ↓
Coarse Decoder
          ↓
Middle Decoder ← Middle Encoded Features
          ↓
Fine Decoder ← Fine Encoded Features
          ↓
Native Reconstruction
```

采用自顶向下的空间恢复。

### 7.2 物理条件

Decoder必须直接使用现有共享描述器生成的目标通道条件。

不增加用户目标输入。

目标通道的物理条件在训练任务建立时从原始观测通道自动获得。

### 7.3 原生输出

必须保留：

- 原生像元数量；
- 原生QA；
- 原生面积映射；
- 目标有效像元；
- 物理量标准化和反标准化约定。

Fine Token对应的内部空间支持由实际分区决定，不能依赖固定2×2输出。

最终预测需要按真实面积关系映射到原生像元。

不允许输出数量增加就宣称空间分辨率提高。

### 7.4 验收

- 三层Decoder均有梯度；
- 目标通道条件正确；
- 不同通道可独立重建；
- 原生输出尺寸准确；
- 隐藏目标不能泄漏；
- 原生面积映射前后语义一致。

## 8. Phase 5：MAE任务

保留A/B1任务定义。

### A任务

隐藏地理区域。

Fine、Middle、Coarse必须遵循相同目标访问权限。

### B1任务

隐藏整波段。

目标波段真实数值不得进入Online Encoder。

目标物理描述允许进入Decoder。

### 任务权重

配置化：

```python
task_weights = {
    "A": 0.7,
    "B1": 0.3,
}
```

只作为初始候选。

实际归约必须对可用任务进行正确归一化。

分别记录：

- A损失；
- B1损失；
- B1隐藏波段数量；
- 各传感器损失；
- 各任务有效目标数；
- 梯度范数。

### MAE测试

必须进行目标污染测试：

将被隐藏像元替换为大数、随机值或NaN，检查Online Encoder和MAE预测是否保持不变。

还要检查跨尺度交互和Fine邻窗交互不会泄漏隐藏目标。

## 9. Phase 6：JEPA

### 9.1 Target Encoder

复制Online Hierarchical Encoder的结构和初始权重。

Target Encoder使用no_grad。

参数只由EMA更新。

### 9.2 Predictor

使用共享主体。

通过尺度投影处理Fine、Middle、Coarse不同维度。

目标位置和尺度信息从现有几何中获得。

不建立三套完整独立Predictor。

### 9.3 三尺度目标

分别预测：

- Fine目标表示；
- Middle目标表示；
- Coarse目标表示。

各尺度单独统计损失。

### 9.4 EMA

实现：

target = momentum * target + (1-momentum) * online

并明确处理：

- 参数；
- Buffer；
- LayerNorm等状态；
- 检查点保存；
- 恢复后的动量状态。

Target Encoder不能在普通optimizer中更新。

### 9.5 JEPA测试

1. Online隐藏区域污染不影响Online输出。
2. Target Encoder可以读取完整目标。
3. Target Encoder没有梯度。
4. EMA数值与公式一致。
5. Predictor输出尺寸和目标对应。
6. 三尺度损失分别有效。
7. 检查点恢复后EMA状态一致。
8. 不同观测之间不存在目标混合。

## 10. Phase 7：Hybrid训练闭环

统一：

L_total = λ_M L_MAE + λ_J L_JEPA + λ_S L_spatial

训练代码必须支持：

- MAE分项损失；
- JEPA分项损失；
- 多尺度空间损失；
- A/B任务权重；
- EMA动量；
- 混合损失权重；
- 梯度裁剪；
- 检查点恢复；
- 完整验证。

不得只记录一个综合loss。

必须记录各项原始损失、加权损失、有效目标数量和共享Encoder梯度范数。

第一版SAM不作为必需功能。

## 11. 工程性能Profiling

目前不设置Epoch耗时硬门槛。

实际测量：

1. 数据加载。
2. MetadataResolver。
3. 条件卷积。
4. Fine聚合。
5. Middle聚合。
6. Coarse聚合。
7. 三个Transformer。
8. 双向跨尺度交互。
9. Hierarchical Decoder。
10. MAE反向传播。
11. Target Encoder前向。
12. JEPA Predictor及反向传播。
13. optimizer update。
14. 完整Epoch和验证路径的无更新回放（真实参数更新的整轮训练仅在另获授权后测量）。

分别记录：

- wall-clock time；
- CUDA时间；
- 峰值allocated显存；
- 峰值reserved显存；
- 激活内存；
- Token数量；
- 注意力边数；
- 原生像元数量；
- 有效目标数量。

必须使用真实最大场景及真实Batch进行容量检查。

不能通过减少成员、裁图、跳过QA或减少原生监督来改善速度。

先使用不执行optimizer.step、EMA更新或持久训练状态变更的前后向回放测量；明确区分实测与估算，optimizer更新成本未授权时标记未测。完整Epoch无更新回放不是训练Epoch实测，不得混写。未来允许的工程更新也须独立预算与授权。本次文档任务不执行这些操作。完成Profiling后，再依据实测讨论优化，不预设20、25或30分钟/Epoch门槛。

## 12. 自动化测试

至少建立以下测试组：

### 几何与Token

- 动态尺度分区；
- 不同GSD；
- 不同L；
- 非整除边界；
- 不同场景尺寸；
- 原生面积交叠；
- 有效支持率；
- Token动态数量；
- 并行聚合；
- 父子索引。

### Transformer

- 三个分支独立梯度；
- Fine邻窗注意力；
- Middle区域注意力；
- Coarse全局注意力；
- 跨尺度双向交互；
- 多场景隔离；
- 无效Token屏蔽。

### Decoder

- 自顶向下数据流；
- 目标物理条件；
- 三尺度梯度；
- 原生输出映射；
- 隐藏值隔离。

### JEPA

- Target初始化；
- no_grad；
- EMA；
- Predictor；
- 多尺度目标；
- 梯度；
- checkpoint恢复。

### 训练

- A/B任务归约；
- 不同有效任务数量；
- 微批次梯度一致性；
- 中断与恢复；
- 运行身份；
- Viewer日志；
- 完整状态恢复。

## 13. 实施顺序与阶段交付

### 第一阶段：动态几何和三尺度Summary

交付：
- 几何计算；
- 三尺度并行聚合；
- Token数据结构；
- 单元测试；
- 计算诊断。

### 第二阶段：三个Transformer和双向交互

交付：
- Fine；
- Middle；
- Coarse；
- Neighbor Attention；
- Regional Attention；
- Global Attention；
- 稀疏跨尺度交互。

### 第三阶段：Hierarchical Decoder和MAE

交付：
- 目标条件传递；
- 层次化重建；
- 原生像元输出；
- A/B1任务；
- 隐藏值隔离测试。

### 第四阶段：JEPA

交付：
- Target Encoder；
- EMA；
- Predictor；
- 多尺度预测；
- 目标访问隔离。

### 第五阶段：联合训练闭环

交付：
- Hybrid Loss；
- 训练循环；
- 验证；
- 检查点；
- 完整Profiling；
- 最终工程报告。

阶段之间必须进行代码审查和测试验收，不允许先堆积全部模块再集中修复。

## 14. 代码质量要求

1. 几何计算与模型计算分离。
2. 不重复构建可缓存几何索引。
3. 不在Forward中执行不必要的CPU/GPU同步。
4. 不使用隐式广播掩盖张量维度错误。
5. 不用固定Token数量代替真实空间计算。
6. 所有有效性和面积权重具有明确语义。
7. 所有跨尺度映射可以追溯到真实空间关系。
8. 不保留无效参数或未使用接口。
9. 每个主要模块具有独立测试。
10. 不为追求目录结构而过度拆分。
11. 不复制旧模型的代码作为新模型后不做语义审查。
12. 重要计算路径必须有形状、梯度及数值正确性检查。

## 15. 文档维护

完成后同步：

- `docs/DESIGN.md`：增加V3-H演进入口；
- `docs/AGENT_ROLES.md`：登记实施职责及交接；
- `README.md`：增加新架构入口、配置和测试命令。

新增的两个V3-H文档负责具体架构和实施规范。

正式训练结果只能在真实实验完成后进入RESULTS.md。

不建立按日期或Agent会话命名的额外维护文档。

## 16. 最终交付清单

Agent必须提供：

1. 实际提交分支和Commit。
2. 新增及修改文件清单。
3. 设计与代码逐项映射表。
4. 三尺度几何计算说明。
5. 三个Transformer实际注意力范围。
6. 双向跨尺度交互的实际索引结构。
7. Decoder原生输出说明。
8. 目标通道物理条件数据流。
9. MAE与JEPA信息权限图。
10. EMA更新与恢复测试结果。
11. 全部自动化测试结果。
12. 完整参数量。
13. 各模块实际Token数量。
14. 峰值显存。
15. 完整计算Profiling。
16. 未解决问题。
17. 是否达到进入正式训练前置门禁的结论。

最终报告必须区分：

- 已实现；
- 已通过测试；
- 已进行真实数据检查；
- 已完成性能测量；
- 尚未验证；
- 需要用户决策。

## 17. 最终验收

只有设计文档规定的全部核心信息流均得到实现，并且数据隔离、几何正确性、梯度、EMA、检查点恢复和计算Profiling通过，才能判定V3-H工程实现完成。

工程验收不等于研究效果验收。

原文档任务不包含正式训练授权；当前用户后续指令已扩展授权。具体训练仍以当前配置、门禁证据和新运行身份绑定，不自动恢复任何旧作业。

## 18. 可执行任务、依赖与门禁

### 本次基线检查与授权

远端与本地主分支HEAD均为 `713cad371d25830226662d297f9a6f26e35efe46`。已读根AGENTS、项目README/DESIGN/AGENT_ROLES/RESULTS、shared_mae配置、几何与模型路径及相关数据/训练/测试接口。原工作树已有未提交DESIGN增量、旧兼容路径下的agent_work.json增量及 `.project/v3_h_design_integration/`；保留原状，不将这些改动自动并入本次提交。

根规则通常只维护五份正文，本次用户明确指定两份V3-H专用MD，作为特定例外；原五份仍保留导览、事实和职责入口。两份专用文件不复制正式结果或建立每日流水账。

已运行只读 `python .project/inspect_workspace.py --root .`。Encoder V3旧active认领覆盖src/tests/outputs，不能凭陈旧状态接管；未来代码阶段必须先核对并显式移交具体路径。纯文档本轮范围仅为本次五份Markdown，协调者统一合并；计划中的角色为待认领职责，未启动多个Agent或任何模型作业。新任务仍使用原 `outputs/encoder_v3/agent_work.json` 登记，不建立第二份运行任务表。

### P0必须先交付的契约

1. 读基线并生成设计—当前实现差异表；核验实际坐标系支持。现有AreaCache针对规则轴对齐槽位，不能直接声称支持任意旋转/多边形。
2. 选择并写出确定性的 `Partition(L, NativeGrid[], scale_policy)` 规则，包含多GSD支持参考、层级尺度、裁边、嵌套/交叠关系、极粗输入和零支持情况。未具化前不得进入P1。若选项改变已确认设计，先取得用户决定。
3. 至少给出L=128m配10/20/30m和各向异性GSD、另一个L、偏移网格、不整除边界的手算分区/面积/Token数；这些是验证样例，不是固定布局配额。
4. 固定Fine邻域、Middle区域及跨尺度调度；记录边数量级，确认Coarse每Patch一个Token的首版策略。
5. 固定每尺度通道融合、Decoder查询、损失归约与EMA状态契约；几何、QA有效性、Online可见权限和Target目标权限分别表示。

### 阶段与Agent任务卡

所有路径相对 `earth-representation/`；下表为后续任务分配模板。一个Agent可兼任工程职责，最终验收者应独立复核关键泄漏与恢复证据。只在明确启动后登记owner和状态；不因文档中出现任务就自动派发。

| 阶段 / 任务ID | 依赖 | 负责人职责与允许范围 | 必交付与退出门禁 |
|---|---|---|---|
| P0 / H00 契约与差异审查 | 当前两份文档 | 协调者：两份V3-H文档；只读shared_mae及原登记 | 上述5项契约、逐项追踪矩阵、范围移交；G0通过后才能编码 |
| P1 / H01 动态几何和并行Token | G0 | 几何Agent：hierarchical下geometry/tokenization及对应测试 | 面积守恒、不同L/GSD/边界手算、QA/可见支持、通道融合、可变数量与padding、共享前端调用计数；G1 |
| P2 / H02 三主干 | G1 | Encoder Agent：hierarchical下encoder及对应测试 | Fine跨相邻Patch可达且禁止远端直接边，Middle区域索引，Coarse同观测全局；各分支梯度；G2a |
| P3 / H03 双向交互 | G2a | Encoder Agent：cross_scale及对应测试 | F→M、M→F、M→C、C→M逐向干预、梯度和无效关系检查；G2b |
| P4 / H04 Decoder及MAE | G2b | Decoder/目标Agent：decoder/objectives/masking及对应测试 | 自顶向下路径、D_output、真实原生面积读出、A/B1可用任务归约；污染与场景隔离；G3 |
| P5 / H05 JEPA | G3 | JEPA Agent：target/predictor/ema及对应测试 | 同观测目标位置匹配，Target无梯度且不在optimizer，初始化/EMA/Buffer/恢复；G4a |
| P6 / H06 Hybrid闭环 | G4a | 集成Agent：hierarchical训练闭环、新配置及scripts入口、对应测试 | 合成数据梯度、损失分项、微批次观测等权、序列化与下一步一致性；G4b；真实更新另行授权 |
| P7 / H07 profiling与审查 | G4b | 性能/验收Agent：独立outputs/encoder_v3/hierarchical证据；只读冻结成员 | 原生最大场景和真实Batch无更新前后向、冷/热缓存、各模块成本、端到端回放、参数/Token/边数；G5 |
| P8 / H08 工程移交 | G5 | 协调者合并文档；验收者只读代码与证据 | 全追踪矩阵、Commit/环境/配置/成员哈希、未测项；工程结论，不签发训练许可 |
| R1 后续研究验证 | 工程验收及新实验授权 | 评价/执行职责另认领 | MAE独立对照→Hybrid增量→冻结Encoder下游/跨地域/留传感器；研究门禁R，不属于当前执行 |

P1→P8严格按依赖交付；独立验收可准备测试样例，但不提前接管其他Agent文件。每阶段一个可复核提交，失败留在该阶段修复，不以后续loss下降补签。任何复用旧门禁的证据都必须重新绑定新Commit，旧shared_mae验收不能直接放行V3-H。

### 验收门禁的最低证据

- **G0 契约**：没有未定义却准备编码的核心策略，样例可手算，任务范围无冲突。
- **G1 几何**：常量场与面积守恒、QA/全无效/分数边界正确；无隐藏固定配额。使用NaN时先安全选择，避免NaN乘零传播。
- **G2 架构**：三分支独立且并行；四个方向均有实际信息路径；稀疏边符合几何。既做允许边的正控制，也做远邻/跨观测/无效Token负控制。
- **G3 权限**：分别污染A隐藏区、B1整波段及JEPA Online隐藏区，比较Online表示和相应预测；合法Target特征允许随完整目标改变。输入隐藏值梯度为零，卷积前隔离覆盖所有与目标相交原生像元。
- **G4 优化与恢复**：所有应训练参数有梯度，Target无梯度；EMA手算与恢复一致。比较保存恢复后“下一步”模型/optimizer/scheduler/EMA/RNG/采样游标/Viewer身份；真实更新未获授权时只用合成小例或状态模拟，不能冒称真实数据恢复已验证。
- **G5 成本**：硬件/软件/精度、Batch/微批次、L/GSD/尺寸/波段/成员哈希、缓存状态、重复次数和同步方法齐全；报告中位数及波动。OOM记录实际边界，不以裁图/减成员/减监督修饰通过。容量不足则修复路径或回到契约审查，不自行截断Token。
- **R 研究**：冻结成员、地理隔离与等预算对照，区分工程、MAE独立增益、Hybrid增量、下游迁移；单次实验不宣称显著性。

## 19. 设计代码追踪矩阵

以下为实施前映射，**所有V3-H新实现/测试/实测状态均待执行**。实际符号在每阶段提交时填写；不能仅填写目录后标为通过。

| 设计ID / 文档节 | 当前基线定位 | 后续改动与证据要求 |
|---|---|---|
| D01 / §3 物理条件 | shared_mae/metadata.py: MetadataResolver、PhysicalChannel；model.py: SharedMemory、ConditionalConvolution | 只读复用、无按卫星参数表、D_conv/D_fusion/D_output路径及目标条件自动解析测试 |
| D02 / §4 动态几何 | geometry.py: metric_geometry、AreaCache；config_shared_mae_v1.json显式L与layout=8 | L/原生几何保留；固定槽位映射另行替换，记录各尺度实际分区、面积和边界测试 |
| D03 / §5 并行生成 | model.py: _patch_features、固定layout展平Patch Embedding | 一份每通道原生32维内容→三个独立面积分支→尺度投影/条件通道融合；调用计数与依赖干预测试 |
| D04 / §6 三主干 | model.py: GlobalTransformer为旧单尺度全局路径 | 三个独立模块及F邻域/M区域/C全局；索引导出、可达性和跨观测负控制 |
| D05 / §6.5 both | 旧模型无三尺度交互 | 四方向稀疏映射、投影、残差、掩膜及逐向干预测试 |
| D06 / §7 Decoder | model.py: 旧全局Decoder/output；AreaMap.native_predictions | C→M→F及原生读出，不复用固定layout输出假设；D_output、尺寸、面积/梯度测试 |
| D07 / §8.1 MAE | model.py: _prepare/available_tasks；test_shared_mae.py污染/混合GSD检查 | A/B1继承隔离语义；三尺度一致权限、观测等权、可用任务权重归一化 |
| D08 / §8.2 JEPA | shared_mae无EMA Target/Predictor | 新共享Predictor/尺度投影，Target停止梯度、EMA/Buffer/目标索引/恢复证据 |
| D09 / §8.3 Hybrid | train.py: batch_loss/update/save/resume；data.py: EpochPlan | 复用语义并适配新状态，分项loss/目标计数/梯度；微批次与尾批/恢复检查 |
| D10 / §9 成本 | 已有scale/precision/check工具只对应旧模型 | 新路径绑定完整身份，模块与端到端无更新profiling；不设Epoch时间阈值 |
| D11 / §11 成功标准 | 旧测试与RESULTS不证明V3-H正确 | 工程门禁报告及独立研究验证分开；保留所有未测/失败项 |

交接模板：任务ID、owner、授权原文范围、父Commit、改动文件/符号、设计ID、配置差异、测试命令与结果、真实数据/性能证据路径及哈希、失败/未测项、退出门禁结论、下一唯一任务。代码实际信息流偏离设计时先修代码或经明确决策修订设计，不能事后改文档掩盖偏差。

## 20. 当前实施与首段训练约定

当前用户授权独立V3-H实现与试验后进入训练，任务登记ID为`encoder-v3-h-implementation-training`。源码工作树为`C:/Users/yzh/.codex/tmp/v3-h-implementation`；旧主工作树未提交正文和登记增量不纳入源码提交。登记保留旧任务身份，当前新hierarchical路径按本轮用户授权使用；未接管旧shared_mae/latent_response路径。

P0的具体尺度、邻域、调度、Decoder查询及损失/EMA规则见设计§14。代码位于`src/encoder_v3/hierarchical`，测试在`tests/encoder_v3/hierarchical`，配置为`configs/config_v3_h_mae_v1.json`，入口为`scripts/run_v3_h.py`。旧元数据、物理Memory、原生数据适配、Epoch成员逻辑、原子保存和TVLP工具只读复用。

首段按MAE独立10 Epoch准备：显式L=128m、64/96/128维、各2层4头、mask=0.55、A/B1=0.7/0.3、λ_J=λ_S=0、随机初始化、Batch32、观测微批次1、FP32、AdamW lr=1e-4、1 Epoch warmup。TRAIN7442/VAL383及原QA/统计不变，每轮4次完整验证；两次完整VAL高于best10%时安全停止。此配置是首版研究候选，不是最优值。Hybrid另用身份和配置比较增量，不在首段静默开启。

工程证据统一存`outputs/encoder_v3/hierarchical/engineering`，记录真实更新与无更新检查。首版最大场景在通道投影处OOM（真实更新0），保留`max_native_v1`失败证据；`max_native_v2`为无更新中止的分块诊断，不能当成通过。后续版本只调整等价分块和重计算，保留全部Token、空间边、原生像元和监督。容量、真实Batch32、固定原生学习、完整状态恢复、逐模块成本与审查未完成前，不签发readiness或启动正式训练。

本页阶段表是职责规划，不表示独立审查已经完成。若本轮仅自审，必须明确标注验收来源，不冒称另一Agent或独立评价。工程通过也不等于表征有效，正式成绩待真实训练与评价后写RESULTS。
