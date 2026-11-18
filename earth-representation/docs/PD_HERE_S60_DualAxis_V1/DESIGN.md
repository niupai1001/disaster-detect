# PD-HERE S60 Dual-Axis ViT v1：60米单层Patch修改规格

状态：PROPOSED / 尚未修改仓库、训练模型或实测GPU。
本文件只规定一个候选。替代旧R3中“每波段Sub-patch空间Token→大展平压缩”的信息组织，不宣布任何重建收益。
代码核对参照：`a0cbf6959d93d58f002667c520b1def4f976816f`。这不是声称已审阅其后的本地或仓库更新。

## 1. 冻结目标

- 只有一种网络Patch：60m×60m，不重叠；不在Patch内部再产生Sub-patch序列。
- 每个Patch全部可见原生波段共同编码为一个空间Token。
- 同一个Patch内，各可见波段分别编码为光谱Token，再做波段Self-Attention。
- 空间分支在同一观测窗口的可见Patch之间Self-Attention。
- 每个Patch内拼接“1个空间Token＋所有可见波段Token”，做一次局部融合Self-Attention，只保留空间行输出。
- 输出仍是一位置一Token；不把整窗口的光谱Token全部展平成一个全局长序列。
- 无Stem、无物理描述器、无Q槽位、无多尺度、无额外融合后全局Block、无原始像元旁路。
- 全局父观测batch=16；核心S2，沿用已经校正并核验的700 TRAIN/300开发VAL及其父/地理隔离成员。

## 2. 原生几何与数据范围

保持同一480m窗口和原点，60m Patch网格为8×8，共P=64。480m是样本读取窗口，不是额外大Patch或网络层级。
采用已有10波段布局（禁止静默加入60m波段）：

| band | 每60m块原生形状 | 像元数 |
|---|---|---:|
| B02/B03/B04/B08 | 6×6 | 每带36 |
| B05/B06/B07/B8A/B11/B12 | 3×3 | 每带9 |

R=4×36+6×9=198。原始数据不插值，不把20m观测复制到10m网格。空间Embedding输入是按固定band和行列顺序拼接的原生向量，不是伪造的6×6×10立方体。

固定band顺序：B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12。
输出偏移：0,36,72,108,117,126,135,171,180,189,198。

Layout必须保存：patch_size_m、window_shape_m、grid_shape、band_keys、band_GSD、native_shapes、offsets、CRS/affine、origin、patch排序、QA规则、hash。删除`subpatch`字段和整除2断言。3×3是正确的完整Patch，不能通过padding成4×4“兼容”旧分支。

输入边界只可使用无效QA padding；有效像元本身必须落在原生对齐索引。不同CRS、旋转/剪切或非整数原生切片不在本轮支持范围，必须拒绝并登记，不能静默重投影。

复用当前已校正的数据身份，不从历史未修正定标清单重新生成。生产启动前绑定实际源清单、定标、QA、窗口和mask哈希；哈希缺失则阻止正式训练。原120m缓存不能被直接当作60m Token缓存；只能从原生值/QA重新切分，或经严格可逆回填后重新切分。

## 3. 可见性先于所有Embedding

A：隐藏区域在两条分支同时移除，不产生其可学习Embedding。
B1：隐藏波段的原始值在两分支均不可读。空间全波段向量该band槽位填0；光谱序列真正移除该band。保留稳定band ID，不重新编号。

空间值向量r[p,R]和有效性q_input[p,R]：q_input=原QA AND 当前band可见；隐藏值/NaN必须用where置零，不用NaN乘0。a[N]表示当前源band可见性。

QA无效不等于训练遮盖。目标QA可作为数据有效性元数据使用，但其像元值不得参与输入、统计或缓存。若某地理位置没有任何可见观测证据，不向Encoder伪造一个有内容的Token；Decoder可使用Mask槽位预测其合法目标。全观测无可见证据则拒绝该视图。

## 4. 空间分支：一个多波段Patch一个Token

输入e_sp[p] = W_val r[p] + W_qa q_input[p] + W_avail a + b，均输出D=128。
值和QA投影都是198→128；a为10→128。同一映射对所有空间位置共享。QA/身份编码不是物理描述器。

仅打包可见p，并加入固定二维地理位置编码（相对该480m窗口，单位为60m）。
V = SpatialBlock(E_sp_visible + PE_geo)。

SpatialBlock：一个标准Pre-LN Transformer，D128、4头、MLP×4、GELU、dropout/drop_path=0、LN eps=1e-6。注意力序列是空间位置，而不是逐band的Sub-patch。保留各p的索引和输出Token数。

## 5. 光谱分支：同一60m支持域的band Token

对每个可见(p,b)：
e_spec[p,b] = Linear_by_shape(flatten(x_visible[p,b])) + LinearQA_by_shape(qa[p,b]) + fixed_band_code[b]。

6×6输入36→128，3×3输入9→128；相同shape共享映射。无内部Sub-patch划分。
固定p，将可见band组成序列，执行一个SpectralBlock（D128、4头、MLP4，其他参数同空间块）。全部可见波段共享此Block。

光谱序列输出为[P_visible,C_visible,128]。不执行Flatten(CD)→D，不设置spectral_reduce。QA无内容band不作Key；全空序列跳过并由上游合法性处理。

## 6. 同位置融合：一次Self-Attention，然后取空间行

对每个可见p：
U[p] = concat_tokens(V[p]+type_sp, S[p,visible_bands]+type_spec)。
每组长度1+C_visible；type向量用两项小型可学习类型编码、初始std0.02，仅区分分支，不代表物理信息。

U'[p] = FusionBlock(U[p])；z[p] = LN(U'[p,0])。
FusionBlock只用一层标准D128、4头、MLP4。保留空间行，其他光谱行不送后端。空间输出仍可经Attention访问各band特征，残差保留空间路径。

禁止：把所有p和band一起全局flatten；另加fusion后跨PatchBlock；再次拼接整个窗口压成一个Token；将融合命名为无损。空间Token已经具有跨Patch上下文，融合读取同位置的光谱内容。本轮不再叠加多尺度或深层上下文网络。

输入索引(p)是一一对应依据，输出不是纯空间或纯光谱语义。光谱、空间分支都使用同一批观测，可能存在冗余，这需要实验评价。

## 7. MAE Decoder

输入GeoEncoding={values[B,Pv,128],ids_keep,valid,grid_shape,layout_hash,band_visible}。
Decoder仅接受该接口和固定结构元数据，不读取原始可见值或未压缩分支缓存。

- Encoder到Decoder：Linear(128,128)。
- 使用唯一mask_token[1,1,128]补齐64个地理槽位；仅A或没有源证据的有效位置需要补Mask。
- 所有有效槽位加入新的8×8二维位置编码；加入decoder_band_availability(10→128)。
- 两个Pre-LN Block，128维4头，MLP4；最终LayerNorm。
- 共享Linear(128,198)，每个Token预测该60m块10band原生数值。
- 按offsets与8×8几何回填；不插值、不PixelShuffle、不面积求积、不逐坐标MLP。

B1隐藏一个band时，其他band仍产生每个p的联合Token；不得把所有p改成Mask。预测完整198维，loss只计算目标band。

保持同名对照的zerohead：prediction.weight=0，prediction.bias按TRAIN band均值填到对应原生像元槽位，随后可学习。其他模块标准初始化。zerohead造成首步上游梯度为0是预期行为，必须验证首步输出权重更新后上游梯度恢复，而不是据首步断定断梯度。

128维直接输出198维仍然是受限表示，不保证任意纹理可恢复。减小单Token覆盖范围不自动消除重复；64个位置的输出仍需实际内容支持。

## 8. A遮盖和同目标比较

本轮主比较沿用旧120m版本已登记的原生隐藏区域。将每个旧隐藏120m块映射成对应2×2个60m Token；同一个480m窗口通常64槽位、48隐藏、16可见。

这个2×2仅是mask索引映射，不参与Embedding或Attention分组，不产生外层大PatchToken。

不得重新独立采样60m位置、保持75%比例后宣称任务完全相同。独立60m随机遮盖可以以后单独登记，本轮不同时开展。

有QA空洞时，旧目标的实际原生pixel IDs为权威；可见/目标支持量、父观测数分别记录，不能为凑固定Token数更改隐藏真值。
B1沿用父观测、epoch和band身份确定的缺带排程，保持与对照同一隐藏band。

## 9. 形状验收（全有效、10band）

| 阶段 | A | B1 |
|---|---|---|
| 可见Patch数 |16|64|
| 空间输入Token/观测 |16|64|
| 光谱序列 |16组×10|64组×9|
| 局部融合序列 |16组×11|64组×10|
| Encoder输出 |[B,16,128]|[B,64,128]|
| Decoder序列 |[B,64,128]|[B,64,128]|
| 预测 |[B,64,198]|[B,64,198]|

A中4个头的空间注意力各16×16；B1中各64×64。融合各组的4个头分别为11×11或10×10，绝不能成为704×704的整窗口融合。

## 10. 训练、性能与停止规则

沿用已校正的700/300开发划分、band、QA、定标、窗口原点与反射率单位；不同时切换数据域、归一化或损失。
MSE按隐藏合法像元→band→父观测归约；paired权重A/B1各0.5，同权重状态计算后一次更新。
AdamW lr1e-4、weight_decay0.04（1D/bias除外）、betas(0.9,0.999)、clip1.0；无PCGrad、结构损失、感知损失和物理调制。继承固定学习率对照，不暗增新的schedule。

全局16父观测；单卡真实批量FP32作为参考。双卡目标8+8，只有实际更快才采用；4×2累积仍是16。尾批按真实父数归约。A/B1不是两次optimizer step。没有用两GPU强行拆A/B、每步CPU numpy广播模型的要求。
按shape组批量处理6×6、3×3光谱Embedding；空间整窗口矩阵操作；所有局部fusion组打包到batch维。可以按两个shape循环，不允许逐位置forward。QA mask正确时使用官方SDPA；BF16和编译单独数值/性能验收。checkpoint重算默认关闭。

执行次序：
1. 工程：新Layout、回填、旧mask复用、QA与隐藏值隔离。
2. 固定16个TRAIN父窗口：A-only/B1-only各200更新；先与本任务同输入基线比较，不要求不可预测A纹理为零。
3. 全新初始化700/300 paired试验：最多30epoch=1320步；第1/5/10/20/30周期审查。第10周期结构无改善且未通过同输入基线检查时停止、登记，不自动延长到200epoch。
4. 质量达到预设可用目标后，再做第二seed及扩大训练。不得用重复尝试VAL选择最优后声称独立TEST结论。

监测物理域MSE/Bias²/CenteredMSE之外，拆分块均值和块内纹理；图像梯度分块内/跨60m边界，旧模型也在同一原生像元与分析网格重算。记录去均值共同模板能量、预测对可见结构打乱的最终输出响应。梯度幅度增加不等于真实纹理恢复。
早停是项目管理决策，不是证明架构不可能收敛。保留所有失败结果。

## 11. 成本（预算而非计时）

C(L,d)=12Ld²+2L²d MAC。谱/空间/融合各1块、Decoder2块，d=128。
A约0.101G MAC/观测；B1约0.292G MAC/观测；16父观测paired约6.284G MAC的主要前向矩阵运算。
预算包括数值与QA投影、可见性投影、Encoder/Decoder线性层和预测头；忽略bias/norm/GELU/softmax、反向、IO、同步与优化器。
不将这些数字直接换成GPU秒数或断言总加速倍率。新版本没有旧的198×128巨型空间压缩，但光谱和融合在每个60m位置执行，仍有成本。

测速：20步热身、50步A/B1/paired分别测P50/P95，记录实际父数、原生输入/目标像元数、QA有效数、显存、kernel/CPU等待及模型模块耗时。必须同窗口同原生目标；旧120m模型在完整480m上与新60m模型比较，不能裁小输入冒充加速。

## 12. 代码迁移清单

对照固定提交a0cbf695中的旧文件，不覆盖旧检查点：
- 新目录建议src/pd_here_s60/，model_id=`PD-HERE-S60-DualAxis-v1`。
- `model.py Layout`：删除subpatch与divisible-by-2；增加真实band shape/offset/patch_m/grid_shape。
- `data.py`：删除120、4×4、16、4*h等硬编码；以60/8×8/P推导，仍用原480m窗口。
- `spatial_values`：从逐band Linear(4,D)改成联合原生向量Linear(198,D)。删除to_subpatch、spatial_reduce。
- `spectral_values`：36→128与9→128按shape共享；删除spectral_reduce，保留bandToken至局部融合。
- `fuse/global_block`：改成跨Patch空间Block＋局部(1+Cv)fusionBlock；不再额外全局叠层。
- `decoder_position/prediction`：8×8地理序列与Linear(128,198)，从头初始化，不按形状凑旧权重。
- `evaluation.py`：unpatchify读取grid_shape；A基线不硬写4/16；OLS基线不能再用h==12判10m，需要真实GSD/shape契约。模型输入仍不重采样；OLS自己的10→20m面积平均是登记的基线逻辑。
- 旧A IDW是按120m块均值计算的基线；同目标主比较保留其原预测方法，再映射到统一目标。新的60m块均值IDW可另报，但不得冒充同一个基线。
- 全部运行绑定新model/layout/hash及复用的原生目标hash。新旧改变不止一个模块，应称整体候选，不做单层因果归功。

## 13. 必须运行的验收（本文件未声称通过）

原生6×6/3×3 patchify-unpatchify；band/像元唯一索引；旧120m mask→60m mask原生集合一致；隐藏值替换与梯度隔离；band身份稳定；QA和padding隔离；一个Patch一空间Token；局部fusion分组和跨父隔离；MAE共享mask与位置恢复；B1不删除有效地理位置；zerohead后上游梯度恢复；loss band/parent权重；单卡/累积/多卡/AMP实际更新；有限值；断点恢复；完整原生目标评价；输入置均值/打乱后的最终输出响应。

工程PASS和重建PASS分别报告。尤其不以总loss下降、Token余弦降低、测试数量或参数量代替结构恢复证据。

## 14. 参考与证据

- 项目旧model.py：固定提交a0cbf695，src/pd_here_geotoken_dualvit_r3/model.py；确认旧Sub-patch、reducers、Layout约束。
- 项目旧data.py/evaluation.py：同提交；确认120/4×4/12像元的硬编码，需要同步迁移。
- 项目dataset_identity.json：同提交，outputs/reports/pd_here_r3/runs/dataset_identity.json；只用于布局与700/300身份范围，不替代后续修正标定发布。
- 项目train_pd_here_r3_texture_study.py：同提交；zerohead与TRAIN均值初始化、固定AdamW来源。
- ViT原文：https://arxiv.org/html/2010.11929v2 。只支持标准Patch投影和Transformer形式，不背书本双分支组合。
- MAE官方：https://github.com/facebookresearch/mae/blob/main/models_mae.py 。只编码可见、共享mask补齐、位置编码及Patch预测头。
- 用户附件《PD-HERE V1 架构审阅与性能诊断报告》§4、§6：冻结成员/QA/定标/原生目标/归约；工程与质量分开；不以过程证据替代表征有效。
