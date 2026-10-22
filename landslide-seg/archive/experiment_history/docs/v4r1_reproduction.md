# v4r1：运行与复现

本轮主线：单期 Sentinel-2 十波段，复用 Hugging Face 官方 SegFormer MiT-B0 编码器与解码器；原网格细节残差分支独立开关。九个训练事件，两个开发事件；seed1234。没有新增粗定位、文本语义、DEM或多时相输入。

## 冻结输入

- 数据缓存继续使用 `data/s2_10band_v4_p20/`，未重写原始nc与历史缓存。
- 新协议：`manifests/v4r1/protocol.json`；新清单：同目录 `pool_{25,50,75,100}.jsonl`。
- 标准化：`manifests/v4r1/stats_core25.json`，仅共同559个训练图块的有效像素拟合；所有组共享。
- 实验配置：`configs/segformer/v4r1/experiments.json` 及六个独立YAML。
- 配置与源码、清单、统计的快照：`outputs/segformer/v4r1_batch/reproducibility_snapshot.zip`，不包含大型影像缓存与权重。
- 历史v4 base已标记弃用，并加入契约校验，避免继续以旧统计启动严格OOD训练。旧checkpoint仍可按原配置重载。

New Zealand曾参与旧统计并被查看冒烟分数，属于有查看历史的后续冻结评估事件。主训练清单不包含它；本轮不推理它。Italy不进入本轮训练或模型选择。两者原有身份与文件均保留。

## 控制规则

每组3600次成功optimizer更新；完整batch64由确定性shuffle样本流跨epoch接续，曝光230400次。全ignore batch不计更新。原epochs上限不截断updates模式。

每100次更新进行同口径验证，共36次。Hiroshima448和Thrissur221分别累积TP/FP/FN，再计算两个事件等权平均IoU选择best；同时保存固定末态last。完整训练集采用无增强推理，不将训练loss当作训练IoU。

last checkpoint保存optimizer、scheduler、global_step、采样epoch/offset、每样本曝光次数、Python/NumPy/Torch/CUDA RNG以及统计值和身份。CPU真实小模型集成测试验证中断恢复与不中断的权重、随机状态、scheduler和采样状态一致；真实SegFormer额外验证输出尺寸、各组梯度和保存重载。

## PowerShell命令

在项目根目录运行，Python为：

```powershell
$py = 'C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe'
```

首次生成修订（本轮已完成；存在冻结文件时拒绝覆盖）：

```powershell
& $py scripts/prepare_v4_revision.py
```

定点测试；基准临时目录应位于项目outputs内的专用路径：

```powershell
& $py -m pytest tests/test_update_training.py tests/test_metrics_pooled.py tests/test_segformer_metrics.py tests/test_segformer_metrics_v2.py -q --basetemp outputs/segformer/v4r1_reproduction_test_tmp
```

运行六个预先声明的实验，自动对best/last做TRAIN/DEV评测和共同训练参考池评测：

```powershell
& $py -u scripts/run_v4_revision.py
& $py scripts/summarize_v4_revision.py
```

如只复现某组，使用独立experiment避免覆盖：

```powershell
& $py scripts/train_segformer.py --config configs/segformer/v4r1/segformer_b0_direct10_detailres_n025_u3600_s1234.yaml --device cuda:1 --set segformer.experiment=reproduction_direct10_detailres_n025
```

在同一配置下恢复中断run：

```powershell
& $py scripts/train_segformer.py --config configs/segformer/v4r1/segformer_b0_direct10_detailres_n025_u3600_s1234.yaml --device cuda:1 --resume
```

单独评DEV：

```powershell
& $py scripts/evaluate_segformer.py --checkpoint outputs/segformer/s2_10band_v4r1_p20/segformer_b0_direct10_detailres_n025_u3600_s1234/1234/best.pt --split val --device cuda:1 --tag best_val --role DEV --save-predictions --figures 4
```

## 指标与解释

- 像素主指标为有效区前景pooled IoU/Dice/Precision/Recall，事件均值另列。
- AP为有效像素精确非插值average precision，同分概率作为一组阈值；不以粗直方图近似。
- 对象采用已有v2口径，8连通、一对一匹配IoU≥0.25，未截断对象为主分母；额外报告1–4、5–9、10–19、20–49、50–199、≥200像素六档。
- 边界F1容差1像素。没有独立核实的负影像时，影像级误报率为N/A，背景像素FPR仍可计算。
- 各档真实训练拟合与共同参考池指标分别保存；共同参考池对小档含未训练样本。
- 单seed是诊断实验，不声称统计显著或跨seed稳定。fixed-update平台不排除曝光不足；历史v3与本轮不构成仅改变数据量的对照。
- 错误图谱仅按数值误差选例，不把模型漏检直接解释为标签错误。

结果表、曲线与案例位于 `reports/v4r1_results/`，自包含报告为 `reports/v4r1_execution_report_2026-10-19.md`；只有六组全部完成后汇总脚本才会写出正式报告。
