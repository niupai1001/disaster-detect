# v4r3：低编码器学习率与 DOFA 系统对照复现

本轮对应 `docs/v4r2_next_actions_2026-10-19.md`，只有两组新正式训练，均为 seed1234。历史 SegFormer 结果直接读取，不重新训练，不改历史 checkpoint。实际结果及完成状态以 `reports/v4r3_execution_report_2026-10-19.md`、各组 `status.json` 和 `reports/v4r3_results/completion_audit.json` 为准。

## 数据与协议

- 保留原十波段顺序 B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12、128×128 缓存网格与有效区，不调整标签或事件划分。
- `manifests/v4r1/pool_100.jsonl`：TRAIN2223、DEV669（Hiroshima448、Thrissur221）；`stats_core25.json` 只拟合559个 TRAIN 样本，各系统共用。
- 3600次成功更新、batch64、230400次样本曝光、每100次完整 TRAIN/DEV 诊断；按两个 DEV 事件 IoU 的算术均值选 best，同时保留 last。
- 同一 masked BCE+Dice、增强、warmup/poly、权重衰减和阈值0.5。对象按已冻结 v2 匹配规则评估，排除跨有效边界截断对象；1–4px 可评对象共77个。
- 本阶段不运行 Italy/New Zealand，不把无目标图块视为经过核实的无滑坡影像；可靠负影像误报率仍为 N/A。

## 两组配置与执行

PowerShell 工作目录为 `N:/Users/yzh/Disater/landslide-seg`；本机解释器为 `C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe`。以下命令是复现入口说明，已有产物无需重跑；复现应使用独立输出目录或独立项目副本，保留原结果。

```powershell
$py = 'C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe'
& $py scripts/run_low_encoder_lr_control.py --validate-only
& $py scripts/run_low_encoder_lr_control.py
& $py scripts/validate_dofa_interface.py
& $py scripts/run_dofa_finetune.py
& $py scripts/summarize_v4r3.py
```

低 LR 配置：`configs/segformer/v4r3/segformer_b0_direct10_detailres_pretrained_encoder_low_lr_n100_u3600_s1234.yaml`。与历史全量 detailres 组相比，唯一训练因素是首层以外 encoder LR 从6e-5降到6e-6；首层保持6e-5，decoder/头/细节分支保持6e-4；3753602参数全部训练，FP32。

DOFA 配置：`configs/segformer/v4r3/dofa_base_upernet_pyramid_binary_n100_u3600_s1234.yaml`。encoder6e-5、随机初始化 neck/head6e-4；BF16 前向，FP32 参数/优化器/损失，周期与最终评测均为 FP32。正式训练从官方预训练权重重新初始化，不使用8样本过拟合 checkpoint。

两组正式运行时使用的源码分别存于 `outputs/segformer/v4r3_low_encoder_lr_batch/training_sources.zip` 和 `outputs/segformer/v4r3_dofa_batch/training_sources.zip`。低 LR 实验先于 DOFA factory 接入；严格复现低 LR 时，在独立项目副本中覆盖其对应源码快照，再运行验证和训练入口。控制脚本会拒绝与原验证源码不符的运行，这是追溯检查，不能删除检查以冒充相同实现。也可用通用 `scripts/train_segformer.py --config <yaml> --device cuda:1` 运行当前源码的新复现，但应记录其新的源码身份，不能覆盖本轮结果。

DOFA 正式入口支持从本组 last 恢复，且检查接口验证、配置和源码哈希。评测命令使用 `scripts/evaluate_segformer.py --checkpoint <best.pt或last.pt> --split train或val --role TRAIN或DEV --device cuda:1 --tag best_val`；各组四项实际命令和日志在 batch 目录，best_val 保存预测、效率与示例。不要使用本例 tag 覆盖另一 split 的历史评测。

## 复用组件与权重身份

- DOFA 官方仓库 `zhu-xlab/DOFA`，commit `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9`；完整源码在 `vendor/dofa-official-0cfb7e1/`，归档身份见 `source_identity.json`。
- 密集编码器直接加载官方 `downstream_tasks/geobench_segmentation/model.py` 与 `wave_dynamic_layer.py`；使用第3/5/7/11层特征，均为8×8、768通道。官方位置编码插值函数把14×14位置网格适配到8×8，未把影像放大。
- 权重 `weights/dofa/XShadow_DOFA_base/DOFA_ViT_base_e100.pth`，HF revision `7a5219e48d2f8848511b0fabea7920a8836bc480`，SHA256 `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1`，与远端 LFS 身份一致。需要下载时使用 `scripts/prepare_dofa_weights.py`，它校验身份而非默默替换。
- 预训练加载没有缺失的编码器键；只忽略官方密集接口不用的 `mask_token`、`norm.weight/bias`、`projector.weight/bias` 五个预训练键。完整加载记录在 `reports/v4r3_dofa_loading_probe.json` 与 run_meta。
- 特征金字塔直接复用 MMSegmentation1.2.2 的 `Feature2Pyramid`，缩放4/2/1/0.5生成32/16/8/4特征；解码头直接复用 Transformers4.57.6 `UperNetHead`，hidden512、一个二值输出头，无辅助分支。随后插值 logits 到128×128。特征放大不代表获得新的观测细节。
- 所有编码器、neck、decoder可训练；仅官方固定位置编码49920参数不可训练。总150547457参数，可训练150497537。没有引入独立原网格 Stem、粗定位、文本模块或新完整视觉编码器。
- 波长来自已安装 TerraTorch1.2.13 的 DOFA 波长表，通过 AST 提取；映射与源文件哈希在 `vendor/dofa_wavelength_mapping_terratorch_1_2_13.json`。单位微米，依序0.49/0.56/0.665/0.705/0.74/0.783/0.832/0.864/1.61/2.20；这是名义波长，未声称按单个平台 SRF 积分校正。保持原有十波段归一化。

## 环境隔离与核验

环境与所有隔离依赖的精确版本见 `reports/v4r3_environment.json`。DOFA 所需 mmengine/mmcv-lite/mmsegmentation 等只安装到 `vendor/dofa_runtime`，不修改原 Python 全局环境；模型按配置插入该目录并直接加载纯 Python neck，避免依赖编译算子。复建隔离依赖时按该 JSON 中的包和版本安装到同名 target，使用 `--no-deps` 保持版本明确，再核验配置内的组件 SHA256。

实际验证包括：5项优化器测试、21项现有适配器/模型/精确更新回归测试；低 LR 真实3步训练与重载；DOFA真实十波段前向、全组有限非零梯度、8样本100步过拟合、重载误差0、batch64反传。过拟合与 batch64 的数值在 `reports/v4r3_dofa_interface_validation.json`，不当作泛化效果。

汇总脚本核对每组自己的源码快照、best/last 文件哈希、数据/统计/协议身份、3600更新与曝光、36次选模、混淆矩阵和对象统计。固定失败案例沿用 v4r1 detailres 的选例 ID，不根据新模型成绩重新挑选。图像可见性/标注时相仍未经逐例人工判定，不用预测差异声称标签有误。

DOFA 的编码器、预训练、decoder、容量、训练精度与 SegFormer 均不同；本轮是完整系统比较，不能据此单独证明波长建模或遥感预训练的因果贡献。单 seed、反复使用同一 DEV，均不能声称独立测试泛化或统计显著。
