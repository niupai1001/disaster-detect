# Encoder V3-H：工程实施与代码验收计划

状态：旧A/B1正式运行保持STOPPED1312/5.632 Epoch，best932/第4轮末/综合383VAL0.4444412768，图集见RESULTS。用户另选随机初始化A-only独立10 Epoch，先完成等价加速与门禁再新冻结启动；旧预算不恢复，Hybrid不启用。当前执行与验收见§24，历史边界保留。设计依据：[V3-H_DESIGN.md](V3-H_DESIGN.md)。

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

正式训练遵循后续用户授权；加速和恢复验收后，用户明确“开始训练”，当前已在新绑定发布中从165次更新续跑，原发布门禁不用于直接放行新源码。

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

以下保留实施前差异映射；实际符号和工程证据见§21。不能仅填写目录或输出形状后标为通过。

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

## 21. 实现与工程证据

首版实现已提交为`9929ca17f84af8d9ad9f6ae581098e2b35879cc0`，Git源文件树为`5a350aa2cf5aa49ec57f7aee45c7a7f908e64f4d`。源码身份为`31da41b2a8b5aa1f78152bbcc904b2168fde0db4e4e06d929d501c8ffbae93b6`。训练发布保存在`outputs/encoder_v3/hierarchical/release_mae_v1`，runtime内167个文件及设计合同单独冻结，加载后身份与最终工程证据相同。

| 设计项 | 实际符号（hierarchical下） | 主要验证 |
|---|---|---|
| D01物理条件 | Encoder.memory/convolution，只读复用shared_mae类 | 元数据回归、共享参数、各分支梯度 |
| D02几何 | geometry.partitions/edges/area_map | 不同L/GSD/各向异性/偏移/分数边界、常量场与原生面积往返 |
| D03并行Token | Encoder.native/fuse/projection | 共享一份原生前端、三尺度直接面积聚合、融合分块输出及梯度等价 |
| D04三主干 | SparseAttention、GlobalBlock、Encoder.branches/coarse | Fine邻窗正控制、远端直接边负控制、Coarse分块全局与整块数值对照、场景隔离 |
| D05双向交互 | Encoder.cross | 四方向逐一源有效性/干预/梯度检查；F→M→C→M→F |
| D06原生Decoder | HierarchicalMAE.decode/output、AreaMap.native | C→M→F梯度、D_output、面积回原生及隐藏值污染检查 |
| D07 MAE | HierarchicalMAE.masks/available_tasks、train.batch_loss | A/B1权限、QA/NaN、单通道、可用任务权重与观测等权/微批次一致 |
| D08 JEPA | target、predictor、predictor_output、update_ema | 初始化、停止梯度、EMA数值/状态、Online污染与Target可响应、三尺度真实样本目标与方差 |
| D09闭环 | train.save/resume/update/main、writer_for | 精确CUDA下一步恢复、Viewer真实恢复身份、TRAIN角色门禁、原EpochPlan复用 |
| D10成本 | check.py、.project/v3_h_profile.py | 最大原生场景、真实Batch32、模块CUDA事件/冷暖缓存；没有Epoch时间阈值 |
| D11验收 | release_mae_v1与engineering证据 | 单元回归、独立审查与正式训练状态分开 |

最终工程证据：

| 证据 | 结果与含义 |
|---|---|
| engineering/unit_tests_v1.xml | 新旧相关38项全部通过；覆盖CPU数值、隔离、梯度、EMA、恢复及Viewer |
| engineering/max_native_v4/report.json | 确定性FP32，最大34,491,294原生通道像元的完整A/B1前后向通过；Fine1,878,600、Middle613,994、Coarse38,626；约36.3/36.9秒，峰值allocated约28.66/23.86GiB；参数未更新 |
| engineering/batch32_v2/report.json | 真实首Batch32，观测等权、微批次1，完整前后向约27.7秒，峰值约1.19GiB；没有裁图或减少目标，参数未更新 |
| engineering/stability_v3/report.json | 8次真实工程更新（含保存恢复后的重放检查），固定8成员损失0.7082→0.5884；保存恢复后下一次更新模型状态逐位一致，不计正式Epoch |
| engineering/module_profile_v1.json | S2/EnMAP各3次A前后向，按模块记录包含重计算的CUDA时间，数据/几何wall-clock与冷暖状态分开；不能累加重叠模块时间冒充总成本 |
| engineering/hybrid_native_v1/report.json | EnMAP/L8/L9/S2代表样本A/B1混合路径无更新前后向通过；A的三尺度JEPA目标数与Target标准差均记录，不证明长期无坍塌或下游有效 |

保留失败轨迹：max_native_v1为OOM，max_native_v2为中止诊断；stability_v1恢复逐位不一致，启用确定性计算后v2/v3通过。不删除这些记录，不将失败或预检称为正式成绩。工程更新总数24（v1/v2/v3各8），正式更新另以formal_run记录。

用户已明确选择“先MAE独立训练10 Epoch，通过后再做Hybrid对照”，并授权一个只读独立审查Agent。首段λ_J=0，Hybrid正式训练未开始；其最大场景容量与长程稳定性仍须针对实际Hybrid发布另验收。旧共享MAE与本段还改变了mask、任务权重和容量，不能把两者差异直接归因为架构单因素；后续Hybrid须匹配本段数据、配置、初始化和评价条件。

首版只读独立审查`/root/v3_h_independent_review`复跑38项CPU测试，额外检查混合GSD/分数偏移隔离与面积，并走通合成数据完整CLI（初始VAL、weighted更新、journal、定期VAL、best/last、真实resume）。原记录为`engineering/independent_review_v1.json`及`independent_review_supplement.json`。后续严格复核发现首版证据未完整覆盖计划要求的元数据契约、损失梯度分项、全部恢复状态及端到端成本，故不能把旧放行等同于完整计划验收，也不能放行后续候选源码。极粗输入（如GSD256m、L128m、单通道1×1）遮盖后可能无原生上下文；扩大准入前须加强任务可用性判断。Viewer跨崩溃恢复可能保留旧Session未checkpoint指标，重复cursor按Session和真实检查点解释。

首版MAE-only的`readiness.json`和用户`authorization.json`只绑定旧冻结身份。正式目录`release_mae_v1/formal_run`、Viewer Run ID `16b8d41b69ec4224bbb01cc252564f23`已安全停止在165次更新（0.70949 Epoch），保存last.pt；初始383VAL损失2.1678077669 z²，step117 best为0.8197413567。暂停预算保留，不修改旧冻结发布及日志。上限仍为总计10 Epoch，不能在更换代码后再追加10 Epoch，也没有完整Epoch或下游有效性结论。

完整Epoch成本尚未实测，后续由正式运行记录；上述模块/批次成本不作为Epoch速度承诺。无独立EnMAP VAL，尚无冻结Encoder下游或跨传感器泛化结论。

## 22. 加速候选的验收与继续顺序

用户明确要求加速后再试验，并明确“不需要整轮完成”。**性能判断允许使用有代表性的同批次前缀，不要求等待完整Epoch，更不预设Epoch耗时门槛。** 性能、容量与研究效果分别验收；短前缀能证明速度收益，不能代替最大成员容量和恢复正确性。

当前等价加速包括共享网格的原生通道分组、几何缓存、有界原始观测预取、小图邻域批量注意力、按激活大小选择重计算、跳过无目标输出，以及单父向下路径删除无效Q/K。保留原L/GSD/几何动态分区、三尺度直接聚合、全部原生像元、监督和邻域范围；没有将向下路径改成多父注意力。嵌套单父下softmax恒为1，V投影、残差和FF仍传递Coarse→Middle→Fine上下文；权重与AdamW动量按有效V参数迁移。

P0中的2/4倍原生尺度、邻域半径、标量条件通道融合和JEPA查询规则是首版候选的具体实现选择，用户明确确认的是both/neighbor/parallel及动态几何，不能事后将所有实现选择写成用户已逐项冻结。当前标量通道融合也不等同于历史四头通道注意力。设计§14列出这些边界；λ_S仍为0，不冒称已实现未定义的空间损失。

| 证据（performance/下） | 实测结论与限制 |
|---|---|
| paired_batch32_v1.json | 同卡真实32观测、A/B1输出/掩膜/Token/边数及梯度配对通过；64任务，2次重复。最大梯度相对L2约4.6e-5；不同传感器和冷暖状态收益不同，不宣称每成员都加速 |
| matched_prefix_summary.json；baseline_full_cycle_v1.json；candidate_full_cycle_v2.json | 同检查点、同前183批/5856观测，无optimizer更新。训练3343.14→2478.52秒（均批18.2686→13.5438秒，1.34885倍）；含3次383成员完整VAL及检查点3598.96→2631.65秒（1.36757倍、减少26.88%）。最大对应批loss差2.14455e-7。两块同型号5090并行且共享CPU/I/O，不是独占同卡端到端测量；同卡配对证据补充支撑 |
| snapshots/candidate_v2 | 固定长前缀对应源码。后续迁移工具/Viewer元数据不在测量热路径；大场景输出头修复另验容量，不把旧前缀当成该路径的测速 |
| unit_tests_final_v1.xml | 47项CPU测试通过，独立审查亦复跑47项通过；包含全模型/optimizer/scheduler/EMA/RNG/游标逐项恢复、实际CLI同身份续跑与跨代码身份迁移、无效NaN和极小logit掩膜安全、100001-token输出头分块前向及features/condition/head梯度与dense对照 |
| capacity_final_v1/report.json、capacity_final_v2/report.json | 最大34,491,294原生通道像元在输出头OOM，保留失败证据；Windows分配器不支持expandable_segments，不能将该环境变量当成修复 |
| capacity_final_v3/report.json | 修复后最大完整A/B1均PASS：31.196/27.518秒，allocated峰值20.444/18.163GiB；Fine/Middle/Coarse仍为1,878,600/613,994/38,626，原生目标仍为20,191,594/24,951,356；参数未更新。该新大图路径成本单独记录，不套用旧前缀加速比 |
| recovery_final_v1/report.json | 当前候选8次真实工程更新，保存恢复后下一步model/optimizer/scheduler/runner/RNG/modes/EMA逐项完全一致；不计正式Epoch |
| all_modules_final_v1.json | 当前身份EnMAP/L8/L9/S2×A/B1×λ_J=0/0.2共16项模块前后向检查PASS，含激活存储、几何/加载、optimizer及EMA成本；每项1次带钩子测量，不作为稳定速度统计、不累加重叠模块时间。临时optimizer/EMA检查后事务恢复，未写入正式训练 |

两条长回放已按用户要求停止，已取得前缀足以支持“实际加速有效”，不再为证明加速重复完整Epoch。候选容量与恢复检查通过并冻结新发布后，用户明确“开始训练”，已新建身份续跑；原Run仍保持STOPPED165。

后续依赖与执行顺序：

1. 已修复大场景输出头激活内存：分块重计算，原生读出与目标不变；最大A/B1及输出/梯度检查通过，不重复长回放。
2. 已补齐当前身份的完整状态下一步恢复、四源模块成本及只读47项回归审查；D01—D11映射及未测边界保留。最大容量以最终v3记录绑定，不继承失败v1/v2。
3. 提交源码/配置/测试/文档并冻结新发布，旧发布和旧Viewer Run保留。通过`prepare_migrated_run`新建身份/Run和lineage，迁移有效权重、AdamW、scheduler、RNG、游标及best/初始检查点；禁止直接修改旧Run执行身份。
4. 按已授权MAE独立总计10 Epoch继续，再按匹配条件开展另身份Hybrid对照。正式评价另记RESULTS；工程回放和少量更新不作为表征收益。

当前执行：加速源码提交`e659cccaca3c77271216921f8540e081a47fbbd5`，源码身份`b527b1ea5a5b426a47af0edbf1f9e71f6ef151ca9e9d6f51db60a7f8d587fb36`，168个依赖文件冻结于`release_mae_accelerated_v1/runtime`。首个0号卡启动在模型装入GPU时出现nvcuda64.dll访问异常，尚无optimizer更新；最小CUDA操作也复现。改用同型号空闲1号卡，唯一执行配置差异为`device=cuda:1`，另以`execution_gpu1/run.py`显式设置默认设备，防止同步或统计回到0号卡。未修改冻结模型源码。

`performance/capacity_gpu1_v2/report.json`完整最大A/B1通过（约36.23/32.68秒，allocated约20.44/18.16GiB）；`recovery_gpu1_v1/report.json`8次工程更新后下一步model/optimizer/scheduler/runner/RNG/modes/EMA逐项一致。数据/QA/成员/模型/目标哈希不变，复用47项代码回归及模块证据；设备差异、包装器和配置哈希记录在`execution_gpu1/execution_manifest.json`，新readiness/authorization只绑定该实际执行身份。此前0号卡启动失败及未显式切默认卡的检查保留，不冒充PASS。

当前正式目录`release_mae_accelerated_v1/execution_gpu1/formal_run`，Viewer Run ID `cd7c5ee4fa22441c844bfb93586c1d2b`，来源Run `16b8d41b69ec4224bbb01cc252564f23`。`migration_lineage.json`记录原last.pt哈希及165次继承更新；best、AdamW有效V动量、scheduler、RNG、采样游标、初始化VAL和最佳指标均继承，新Run从166步写入真实更新。总预算2330更新（10 Epoch），不是另加10 Epoch；L128、FP32、Batch32、A/B1权重及每轮4次完整VAL和回退停止条件不变。实时进度以status.json/updates.jsonl及真实checkpoint为准，当前运行不等于已完成预算或证明下游有效。

## 23. CUDA异常后的双卡续段

上一段已完成208更新（0.894 Epoch），计算209批时CUDA非法访存退出；保存的last在200、best在175（完整383VAL损失0.6147162990），不是已完成首轮。旧状态、曲线、日志和权重保留。用户明确“开始训练，如果能使用双卡则使用双卡训练”，因此本段允许门禁后恢复并优先双卡，不改变MAE/Hybrid顺序。

Windows当前无NCCL，现有多进程入口仍拒绝未经同步的WORLD_SIZE>1。新增`parallel.TwoDeviceBatch`使用单进程两副本：主卡cuda:1、副卡cuda:0，原Batch32按原生像元成本分工；每观测内部A/B1可用任务权重归一，shard梯度除实际全局观测数（尾批18），两卡梯度求和后主卡统一clip、一次AdamW、一次scheduler、一次EMA并广播全部参数/buffer。两个worker拥有独立数据适配器/EnMAP输入LRU/catalog及模型几何缓存；每批两边清梯度，等待两worker和两卡完成后才更新/保存，失败不应用本批optimizer。完整VAL与梯度诊断仍在主卡，两卡此时无并发训练。

| 门禁证据（performance/） | 结果及边界 |
|---|---|
| failed_batch209_sync_v1.json | 原失败批32观测全部64个A/B1在同步CUDA下前后向PASS，无optimizer更新；不能证明异步错误根因已修复 |
| dual_cpu_tests_v2.xml | 50项相关回归PASS，包含双shard全局loss/梯度、尾批/空shard、EMA广播、worker失败禁止更新、完整下一步恢复、新身份迁移与重放记账 |
| dual_gate_v1/report.json | 真实201/209两批各2次单/双卡配对，loss差0、梯度相对L2约1.30—1.45e-7；吞吐1.85—2.27倍，中位1.941倍。同步CUDA、FP32/确定性/TF32关闭，短段非整Epoch速度 |
| 同上：capacity | 34,491,294和22,307,028原生通道像元两个最大观测同时A/B1前后向PASS，峰值allocated主卡约20.69、副卡约19.34GiB，保留原Token/原生像元/目标；该冷几何容量检查耗时约128.8秒，不套用普通批次速度 |
| 同上：recovery | 3次独立工程更新（含重放）后，双卡自身保存恢复的下一步model/optimizer/scheduler/runner/RNG/modes/EMA完全一致，副本权重精确同步，不计正式Epoch |
| dual_gate_v1/final_source_delta.json | 测量后仅增加两适配器身份检查及未参与热路径的续段迁移helper；重建测量源码SHA逐项匹配，单独绑定最终源码，不混淆旧报告身份 |

配置`config_v3_h_dual_v1.json`只在上一段MAE配置上增加devices、每批checkpoint；科学参数和Epoch计划不变。正式启动沿用实测的`CUDA_LAUNCH_BLOCKING=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，不以同步门禁签发异步运行。旧0号卡初始化故障当时真实复现，目前两卡基础CUDA及双卡真实门禁通过；不擅自重启驱动或其他进程。

继续顺序为：当前代码/配置/测试/文档提交→独立冻结`release_mae_dual_v1`→`prepare_execution_continuation`迁移last200与best175到新身份/新Viewer Run→正式入口核对绑定并reconcile复制的日志→确认实际更新越过原失败209批。原201—208日志在恢复证据中留存，原Run不改；spent_updates保留原208次durable尝试，重放8次成本不被抹去。终点仍为2330个逻辑批次/10个完整数据Epoch，完整覆盖与额外恢复成本分开报告，不因异常重启另加10 Epoch。若再次发生CUDA异常，不在失效CUDA上下文中盲目重试。

双卡正式启动已执行：源码提交`8372d7aabf9ec25af35181fda1bb90a2689cca95`、Git tree `149b5c5cfb0ce630a364f133daccaaf3e14dd92a`，170个依赖文件冻结，源码身份`262013c7aface18cfdee14398cf9fb065a62a93915b855a2cc018428df365b90`。正式目录`release_mae_dual_v1/formal_run`、Viewer Run `3666b4153ea5457697c9b3188e8b5958`已从检查点200写入201步真实更新（spent209）。首次loss0.12524238954283645与原对应批一致；每批检查点和恢复尝试归档正常。最新运行状态、实际cursor与原失败点复核见`training_started.json`及formal_run/status.json，不把启动通过写成全部10 Epoch完成。

当前停止边界：用户明确“暂停当前实验……给我看看可视化的结果”，已请求STOP并保存last1312，状态STOPPED、5.632088Epoch、spent1320；best932/4Epoch，最近一次完整VAL为1282步0.4509382418，不将其归给未重评的last1312。pause_record.json记录检查点SHA与用户暂停指令，保留STOP_REQUESTED；历史authorization不能越过最新暂停指令。可视化仅从冻结最佳权重推理，完整383VAL复算精确一致，参数及原best/last/updates哈希不变。reconstruction_best_v1保留初版证据，v2补充显示截断及验证后段放大说明；无optimizer更新，不作为恢复训练。

## 24. A-only独立消融与加速验收

用户明确“暂停B1训练，仅进行A，优化好加速后开始”，并选择随机初始化、A-only独立10 Epoch。新任务encoder-v3-h-a-only-training，新配置config_v3_h_a_only_v1.json，新发布release_a_only_v1；不加载旧best/last，不继承旧Run/optimizer/游标或8次重放。全局Batch32、每轮7442观测/233批（尾批18）、10轮上限2330更新，VAL383/每轮4次，L128、mask0.55、FP32、lr/warmup/clip及原QA/归一化不变。A=1、B1=0、JEPA=0，零权重任务在train/validate/gradient_contributions/check中统一过滤；可视化也仅评估启用任务。无有效A任务的成员fail closed，旧日志已核验全部7442TRAIN/383VAL具有A，不筛掉高损失成员。

等价工程优化：Cartesian分区直接用轴区间生成每4096 query的稀疏边，避免百万点cKDTree Python邻居列表与全局边排序。候选区间覆盖浮点减法消去误差，最终仍对原归一化坐标应用abs<=1，k保持原升序；generic parent/child sparse_edges仍排序。测试中发现初版一ULP范围不足，已修复并逐边通过。Fine/Middle/Coarse数量、原生面积、边/掩膜/目标和模型参数结构不改变。

双卡完整验证沿用独立VAL适配器，在worker内关闭梯度，按全局成员数求均值，join和CUDA同步后恢复主模型/optimizer/scheduler/RNG/模式并broadcast恢复副本；不是两个shard均值直接平均。异步模式CUDA_LAUNCH_BLOCKING=0单独验收后使用，CUBLAS_WORKSPACE_CONFIG=:4096:8、确定性与TF32关闭不变，每批checkpoint保留。旧异步故障根因未确诊，本次门禁仅证明当前A-only路径在所测范围通过，不声称驱动已根治。

| 证据（outputs/encoder_v3/hierarchical/a_only/） | 结果与范围 |
|---|---|
| unit_tests_v2.xml | 55项CPU回归及独立审查复跑通过，含小数GSD/nextafter半径/>100k CSR与旧树逐边一致、B1零前向、双卡VAL线程no-grad及全状态恢复 |
| paired_async_v1/report.json | 四来源代表及最大34,491,294原生通道像元，旧A-only模型/几何与新A-only同初始化、同输入/掩膜/目标/边/损失/梯度通过；最大loss差0/gradient差0。最大冷A前后向59.65→44.14秒（1.35倍），热30.90→30.95秒无收益；普通热态0.93—1.17倍有波动，不能宣称全部更快 |
| pipeline_async_v1/report.json | 25真实Batch32短回放含旧失败209、两个极大场景与后继通过，无更新回放中位约4.28秒；3次独立工程更新（含重放）的下一步model/optimizer/scheduler/runner/RNG/modes/EMA完全一致。单/双383VAL 23.87→16.65秒、metric2.1634566077精确一致；该指标来自工程副本，不是正式初始化分数 |
| dual_capacity_async_v1.json | 双卡同时两个最大观测34,491,294/22,307,028像元仅A前后向PASS，约50.27秒，峰值allocated主约20.40/副约19.26GiB；无optimizer更新 |
| color_learning_probe_v1/report.json | 随机初值固定四源TRAIN中位原生规模观测、32次独立工程更新，A均值0.5864→0.2182；所选RGB物理MAE下降，但L8全波段MSE0.1537→0.1860未改善。仅说明可学习性，不冒充已消除色差；正式初始化重新从相同seed17开始，不携带工程更新 |

取消B1的减负单列：相同候选代码/同Batch32，AB→A-only约11.58→4.89秒及6.78→4.46秒；这是任务工作量变化，不计入等价代码优化倍数。当前已有前两完整Epoch实测17.05/18.95分钟，含验证、保存及日志等，详见RESULTS和a_only/reconstruction_node_525_v2/timing.json；不预设时间门槛。可训练参数和模型信息流不变，不能仅因参数少而保证高吞吐。

顺序：提交当前源码/配置/测试和文档→冻结新身份并核验全部门禁绑定→签发本次用户明确授权（随机初始化A-only10 Epoch）→新Run不带resume启动→确认初始化383VAL和真实更新、每批checkpoint及A-only日志。旧Run保持STOPPED1312及原SHA。A-only VAL与旧综合0.4444不是同一指标，旧best932的A单项0.4156507235仅作相同固定面板开发参照。后续以A重建和RGB/光谱误差检查质量，不仅看平台，不静默启用B1或Hybrid。

正式启动已执行：源码提交`8172965077e4b9f4d788dd4e7252da8cb4dd9a4c`、tree `1a163e4c4a59434bd27b2a61583da40a16fd996f`，release_a_only_v1/runtime冻结172文件，代码身份`332f0705bccc9fc6cb4baedfb6a2ebb2fe353ccf6d21b2143fcd34b40b5b90c5`。新Run `65525b250ad34747a6024a2cea636754`，从cursor0/spent0、不带resume启动；随机初始best0参数SHA `0958271abbe57a2c505f3716cbdcd1291e253e61d0880d430c43396678b84057`与门禁seed17一致。完整383VAL初值2.1474302937264502（A-only），已核验前28步只有A任务且checkpoint落盘；数据计划和实际状态见formal_run，不能把本启动前缀当作一轮实测或质量改善结论。

第525步验证节点快照：完整383VAL A-MSE0.4123939128、2.2537 Epoch，图集a_only/reconstruction_node_525_v2从不可变selected.pt与对应官方validation_525记录生成，选中VAL案例再次推理相符。新增live-source可视化只修改工作树入口，不改冻结训练runtime、参数或optimizer；训练继续，旧AB保持STOPPED。隐藏区仍平滑、光谱有偏差，不能把损失下降称为色差已解决；本段尚无完整算子级CPU/GPU耗时分解。

## 25. 无物理描述器短段实施

用户明确要求考虑计算效率并暂时训练无描述器架构。顺序：安全停止并哈希保存旧A-only704 → 共享普通卷积及显式TRAIN波段Embedding实现 → 验证无物理条件路径、隐藏目标隔离、同网格合批卷积值/梯度、未知键拒绝及短段状态 → 真实双卡最大容量/单双梯度/恢复/速度和固定TRAIN可学习性 → 冻结代码/配置/成员身份 → 按本次授权随机初始化启动一个完整Epoch → 比较完整383VAL的可见均值基线与RGB/光谱、波段区分及实测成本。

58项CPU测试及独立只读审查通过；267仪器/波段键恰来自TRAIN，VAL全部覆盖。配置epochs10只保持2330更新的原学习率日程，max_updates233才是本段绝对预算上限；退出TRIAL_COMPLETE，不能解释为完成十轮或自动继续。保留Batch32、FP32、每批checkpoint及每轮4次完整VAL，A-only、不加载旧权重，不启用B1/Hybrid。

质量门禁不能只比较固定TRAIN通道均值；需报告合法的可见通道均值基线，区分工程学习和正式VAL，关注跨波段归一化输出是否仍几乎相同。速度用同批同设备的原描述器A-only对照，记录原生Token/目标、并发条件、冷/热态与峰值显存；不依据参数量承诺加速、不用最大场景数字概括普通批次。

本短段已启动：源码0ff7831d07b3aebd45875a17a8d6f7356090750b，174文件冻结，运行目录release_descriptor_free_v1/formal_run，Viewer dde013a749f5431abb20aea2a587eb6d。全部门禁identity与冻结运行相同，随机初始参数cf3da97b…、初始383VAL2.1762696610及前5次真实更新已核验。engineering权重未加载，旧A-only704与AB1312保持停止。现已完成max_updates233并标记TRIAL_COMPLETE，1Epoch完整覆盖；末态VAL0.468964仍差于可见均值0.207968，整轮19.639分钟，结果不构成质量通过或整轮加速。旧运行保持停止，后续预算依用户指令决定。

## 26. 简洁Decoder实施与30轮训练

用户最新授权覆盖简洁Decoder实现、实际加速、L9排查及门禁后恢复EnMAP的30 Epoch正式训练，预热50更新。规格见V3-H_DESIGN§16；主Agent实施与执行，已有独立Agent只读审查。旧发布、旧权重及暂停预算不续写。

| 顺序/依赖 | 实际任务与代码映射 | 门禁/状态 |
|---|---|---|
| D0，已完成 | 完整383VAL合法线性基线、7篇一手文献、L9输入/条件/输出定位 | IDW只读keep；L9旧条件近坍缩已证实，未发现524 L9/1522 L8统计/定标/索引错；不将单次排EnMAP解释为完全无影响 |
| D1，依赖D0，已实现 | simple_decoder.py：16维投影/8邻点传播、46维条件、4系数双线性、精确交叠、可见IDW+变化幅度残差；model.py分派；train.py配置/50warmup/基线与来源日志 | 旧路径默认兼容；QA/hidden污染、flat/one/none、跨Fine积分、实际新CLI、L9波段区分、上游梯度/参数变化；完整CPU72项通过 |
| D2，依赖D1，已完成工程学习 | 5个固定TRAIN，前64步随机原Encoder冻结、后64步全模型并以50步升至1e-4；Decoder前段1e-3，仅工程副本 | 均值降低，L8/EnMAP/纹理S2均降低；全部病例退化≤1%+1e-7，平坦实际仅+1.26e-8；真实L9 .001022879→.001021005、系数span .31877；不声称所有样本严格下降或验证泛化。learning_v1–3失败保留，v4通过；quality_final_v2用最终代码重算 |
| D3，依赖最终D1，已完成GPU门禁 | mask_sums/QA union/valid紧凑执行/同mask多band合批；最终TF32对照、非零头双卡最大容量、完整恢复、原生383VAL基线及同Batch32/失败批后继回放 | pipeline_v1、precision_final_v2绑定最终identity；32/18尾批单/双梯度一致，两个最大观测通过，下一步全状态一致。约1.164倍实际同批提速，不冒称整轮或成倍加速；BF16及高显存调度失败候选拒绝 |
| D4，依赖D2/D3与独立审查 | v3_h_simple_decoder_release.py核对最终身份/成员/文件SHA/初始化，冻结release_simple_decoder_v1，再按本次授权启动新Run | 原物理Encoder初始化与原版一致、新Decoder零末层；不载任何工程/旧检查点；数据7442=233批尾18，30轮6990更新；预热50，双卡Batch32，A1/B0/JEPA0；旧运行不恢复 |
| D5，依赖正式启动 | 检查初始383VAL、真实optimizer前缀、每批checkpoint/来源日志/EnMAP实际输入；按固定验证节点出图并报告IDW增量 | 初值IDW约.144189是构造而不是训练成绩。新run日志为准；L9、EnMAP只有TRAIN来源日志/诊断，不伪称独立VAL；30轮完成和泛化改善都须后续实际结果 |

实现必须逐项贴合：mask_sums与occupied_regions对照原逐band规则；NativeResidualDecoder.context仅合法源；visible_condition仅keep统计；coefficients为7316参数的小头；integrate按真实矩形足迹；forward同mask合批且通道均值保留；warmup_updates优先50个optimizer更新；TwoDeviceBatch按全局观测归约，完整状态保存恢复；writer分别登记原生IDW与S2/L8/L9/EnMAP真实来源，缺来源不补假值。每项以函数、测试、绑定报告核对，不能用“代码完成”代替训练就绪。

正式质量持续比较整体/S2/L8的IDW相对收益、物理/光谱及隐藏边界；近零误差病例不用浮点严格下降作唯一标准。相同固定面板已用于开发，不作未接触测试集。32批测速不能外推精确Epoch时长，待整轮实测；30 Epoch为当前训练上限，训练异常或两次验证显著恶化按现有运行控制停止并保留检查点。
