# GeoToken–DualViT R3：训练、并行与Agent交接

## 1. 数据7:3的精确定义

仅核心Sentinel-2、同一已核验地表反射率产品与QA流程。TRAIN:development VAL=0.7:0.3。先绑定总池，再按父观测/地理隔离组拆分，后切窗口。重复观测、重叠区域及应隔离的多时相样本同组；组隔离优先于机械凑比例。

不默认旧“1000 TRAIN+100 VAL”仍有效，不把它改名为7:3。总池若绑定为1000，则名义700/300；若总池不是1000，按实际组数/父数记录计数。首版优先复用用户已冻结的S2候选总池；不私自并入旧VAL或锁定TEST。若旧验证被用于研发，应标为development VAL，不能声称重新变成独立测试。

必须生成dataset_identity.json：原始发布hash、总池hash、group规则、train/val列表hash、band layout、值域/QA、窗口规划、可用样本排除及理由。绝不造填hash。统计、OLS基线、学习率校准仅使用TRAIN。

## 2. 窗口与任务身份

每父观测每次曝光使用一个登记480m窗口；不足时合法padding，不复制观测值凑面积。初轮固定窗口面板；后续窗口轮换规则预登记。每个窗口均保持同一原生几何/输出索引，不能因为B1隐藏最细band而重建更粗地理网格。

任务mask由(parent_id,window_id,task,exposure,seed)决定。A75%整地理Patch跨所有band共同隐藏；B1整band缺失，两视图独立。QA无效不作为训练目标。任何窗口没有任务合法上下文/隐藏目标时明确拒绝/重新采样并记录，不能计0loss。旧A隐藏80m/其他范围与新120m不同，基线和指标需在新共同支持重做。

## 3. 目标和归约

首轮使用native reflectance MSE，不做真值逐Patch标准化，不加梯度/边缘/感知项。一个父观测的loss：对每个目标band在所有合法隐藏像元上平均，再对有目标的band平均。再对父观测平均；不是将不同像元数band全部拼接后平均，也不是每patch等权改变QA较少patch权重。

paired损失：L=0.5*mean_parent(L_A)+0.5*mean_parent(L_B1)。所有梯度在同一参数状态计算，最后仅一次optimizer.step。两任务的0.5权重不意味着前后向次数减半。

## 4. 初始优化配置与判因顺序

AdamW lr=1e-4，betas=(0.9,0.999)，weight_decay=0.04；bias/LayerNorm不衰减；全局grad clip=1。无PCGrad/CAGrad、自定义位移投影和回溯。固定seed17，各实验从完全相同的初始化开始；旧架构checkpoint不兼容，不作等价续训。

学习率只是第一试验值，不预设最优。若易解TRAIN也无法下降，只允许一次记录在案的TRAIN-only短预算lr诊断：3e-5/1e-4/3e-4各50步，其余不变；仍失败则停，不不断叠加模块或延长训练。

## 5. 最小执行阶梯（不是一次并行展开所有消融）

| 阶段 | 内容 | 预算/规则 | 放行证据 |
|---|---|---|---|
| G0 | 身份、patchify/unpatchify、QA、A/B1隔离、mask恢复、成本 | 不做真实训练 | 工程命令、真实测试、固定manifest |
| G1 | batch16 FP32参考、加速和执行等价 | 20热身＋50计时步，独立丢弃更新副本 | P50/P95、显存、实际Token数、单/双卡权重和更新 |
| G2a | 固定16 TRAIN父观测的A-only、B1-only | 每任务200步，先A后B；各自初始化一致 | TRAIN结构变化、简单基线、误差分解、实际更新 |
| G2b | 同16父观测的paired | 前两项有学习迹象后200步 | 两任务各200次曝光；joint多算力另报 |
| G3 | 新初始化、完整新TRAIN池 | 最多5周期，warmup1周期后cosine到1e-5 | 每周期固定VAL；第1/3/5审查 |

1000总池→700 TRAIN时，全局16含尾批每周期44步，5周期220步；这是计数例子，不硬编码。若总池或分组计数不同，steps=ceil(actual_train/16)。父观测周期与完整像元覆盖不是同义。

G2不是要求不可预测的A细节误差为0。用常量/仿射易解构造和相同合法输入的简单参照界定可达程度。Decoder free-latent上限诊断只在怀疑压缩/输出头时使用，独立标识且权重丢弃，不能当作Encoder训练成绩。

不自动开始更大池或多源训练。未过质量门槛可以保留解释性实验，但必须标DIAGNOSTIC，不能标SELECTED或RUN_READY。

## 6. 基线和指标

A：同band可见值的常数/最近邻/IDW；B1：同TRAIN拟合的光谱线性回归。只使用实际可见输入，在同一native目标和共同QA上评分。跨GSD baseline需要对齐时，映射规则独立登记，不能使用隐藏目标进行校准；它是参照模型的计算，不改变学习模型的原生输入。

按父/波段报告MSE、MAE、Bias²、CenteredMSE；原生相邻像元梯度MSE、幅度比、相关性及高纹理子集。邻接对必须是真实原生相邻且双方有效，不能在展平vector跨行或跨band构造假邻接。跨地理patch边界恢复原生网格后再计算。高纹理由真值定义只作评价，不传给Encoder。

选择候选按预注册次序CenteredMSE→gradient MSE→MSE，同时检查A/B各自无明显退化。相比简单基线持续显著更差时不宣称重建可用。若要加数值门槛，应在看结果前登记为工程目标，不把5%/10%写成学术保证。统计按独立地理组bootstrap，不能把像元当独立样本。已用于开发的VAL不写成锁定TEST。

重建通过后另做冻结Encoder下游probe，不能用重建改善直接证明灾害识别收益。

## 7. 全局batch16和真正批量化

全局16个独立父观测；双卡各8，或每卡4、累积2；单卡16或等价累积。每个父观测A/B1各一个视图，不把视图数32称为batch32。

光谱按(batch*visible_patch)打包；空间把相同原生Sub-patch数的可见(p,b)堆在batch轴；几何仅CPU预计算或低频调用；压缩矩阵批量GEMM；跨PatchAttention每父独立。允许几个shape/band组循环，不允许逐pixel/逐Patch forward。

B1的不同隐藏band模式可分桶/分组。每个band缺失时不计算它的投影/局部Attention；压缩前只用固定零缓存保持shape。各band的出现频率按排程均衡；组桶不得静默重采样改变曝光。

空间/光谱大flatten投影为本轮实质参数，不因“仅一层”忽略其代价。缓存只包括原始值、QA、几何和固定PE，不能跨更新复用学习特征。

## 8. DDP与尾批的准确归约

Linux上优先每卡常驻一个DDP进程。NCCL不可假设在原生Windows可用；环境绑定后实测后端。没有通过多卡等价，先用单卡等价batch16，不用每步CPU numpy搬模型状态的手工流程作为默认。

DDP会平均rank梯度，因此对有效全局父数B_total、world_size=W，rank r应反传：
L_r = (W/B_total) * sum_{i in rank r}(0.5 L_Ai + 0.5 L_Bi)。
这样rank均值就是全局父均值。microbatch也按这个固定全局分母，不对每个microbatch单独取均值再相加。A/B同一模型权重累计，只更新一次。

初版优先将两视图封装在一次DDP外层forward中，合成单个标量反传，以避免unused参数和多次forward的同步歧义。显存需要时，再验证no_sync累积版本。B1-only可能跳过某一native-shape投影组或mask_token，find_unused_parameters按实际路径配置并单独测试，不能假设所有任务所有参数均有梯度。

尾批不重复父观测凑16；按真实全局计数缩放。若某rank无有效样本，需支持零权重dummy且执行兼容collective，或将该尾批单列经验证处理；不得某rank跳过allreduce。验证指标也按实际有效父数合并，不平均rank均值。

## 9. 加速顺序与证据

1. FP32真正batch16参考；2. 数据预取/固定shape桶；3. SDPA；4. BF16 autocast、FP32loss与optimizer状态；5. DDP；6. 最后再考虑compile。checkpoint重算默认关。

分开报告A、B1、paired的encoder/decoder/IO/sync/optimizer时间，P50/P95、peak allocated/reserved、父观测/s、原生像元/s、GPU秒与墙钟秒。profiling用同16父窗口/同mask；编译warmup和完整验证时间另列。

首轮20热身、50测量更新仅作工程profile；profile权重不得接着进入正式训练。每种加速同初始化输出/损失/梯度/实际更新/恢复做对照，FP32 tight容差、BF16容差在看真实结果前用参考范围登记。不存在统一保证1e-6的低精度阈值。

不预报GPU秒数；04成本是MAC而非耗时。新结构短序列可能让DDP通信占比增大，单卡更快时应如实选择，不能强制宣称双卡2倍。

## 10. Agent工作包与验收

Agent A：Layout/原生数据入口、7:3组隔离清单、mask、两路raw Embedding、局部分支、固定槽位压缩、单GeoToken、跨Patch一层ViT、几何回填契约。
Agent B：MAE地理mask恢复、B1可见band条件、完整Patch预测、loss/指标/基线、batch16和DDP、运行身份、成本报告。
双方：共同确认模型返回GeoEncoding与输出offsets，不各自生成不同Layout。旧代码/权重/结果只读；实现放入新命名空间pd_here_geotoken_dualvit_r3。

必测：A/B1隐藏值和梯度隔离；Stem/Descriptor不存在；每GeoPatch唯一Token；B1固定band索引；A共享Mask且补全位置；B1不抹掉可见GeoToken；partial QA和全空拒绝；原生像元与边界回填；loss逐像元/波段/父归约；无跨父Attention；single/multi/accumulation更新等价；可重复恢复；真实GPU峰值。

所有报告区分PASS、FAIL、NOT_RUN、BLOCKED。reference的17项CPU测试仅验证小张量shape/隔离/计算，不代替生产几何、真实数据或训练结果。

## 11. 停止与下一步规则

泄漏、坐标错位、划分跨组、梯度归约错立即停止。基本可达构造仍学不会时查实现/优化/压缩，不开始长训。单任务可学但joint退化才进入匹配曝光的优化诊断。训练可学验证差，先查覆盖/正则/泛化，不直接指认架构错误。

若仅总MSE变小而结构恶化，不能凭“还未收敛”自动加预算。若当前输出仿射低秩限制成立或attention不值成本，单独提出最小后续变更，不把本轮重新改成多种开放选项。
