# R3 当前实现与实验结果

本次归档保留实现与结果，**重复纹理与过平滑问题仍未解决**。最新 `band_tokens256` 完成30 epoch、1320次更新，未通过预设候选选择规则；对照仍保留 `fusion256_global2_zerohead`。单seed开发实验，不代表下游灾害识别收益。

## 当前实现

空间分支对每波段完整原生地理Patch直接线性编码为128维，不使用2×2 sub-patch或空间分支自注意力。光谱分支在波段聚合前保留各波段128维Token。按(父观测,地理Patch,波段)对齐拼接，执行256→256投影；两层全局ViT跨同父观测的全部可见Patch×波段Token注意力。完整4×4地理窗口中，A为40个可见Token，B1为144个；Decoder恢复160个槽位。Decoder宽128、深2，按原生网格形状共享输出头，并有逐波段均值偏置。输出保持[B,16,792]。Encoder与Decoder均有变化，本轮结果不能归因于单个改动。

## 数据、训练与评价

固定700 TRAIN / 300开发VAL，固定窗口，10个S2原生波段，seed17、global batch16、单卡FP32。每组重新初始化，AdamW lr=1e-4、weight_decay=0.04，paired A/B1目标，30epoch=1320更新；以下为last，不是best。原始来源审计和标定修正见[审计摘要](source_audit_summary.json)，修正数据身份见[runs/texture_study/dataset_identity.json](runs/texture_study/dataset_identity.json)。旧G2及10epoch结果使用旧标定数据，只作历史追溯，不与修正后结果作同条件比较。

## 最近完整对照：第30 epoch VAL MSE

| 组别 | A MSE | B1 MSE |
|---|---:|---:|
| decoder128_zerohead | 0.0028203139054568988 | 0.0016763435375026635 |
| fusion256_global2_zerohead | 0.002845882724681099 | 0.0016343693289772652 |
| band_tokens256 | 0.0029207491276265164 | 0.00240892348072281 |

[完整比较和有效计数](runs/texture_study/band_tokens_comparison/comparison.json) · [训练曲线](runs/texture_study/band_tokens_comparison/curves.png) · [A固定样本预测](runs/texture_study/band_tokens_comparison/A_predictions.png) · [B1固定样本预测](runs/texture_study/band_tokens_comparison/B1_predictions.png)

## 机制诊断与解释

共同位置、波段与可见性编码会提高原始Token余弦相似度；余弦高不能单独证明Token完全相同。已保留分层特征响应、干预、纹理RMS和共同模板指标。线性可读性probe提示分支压缩及全局融合后细节可读性下降；它只衡量线性读出，不等价于信息被彻底删除。

[分支层级诊断](runs/texture_study/branch_sublayer_audit/report.json) · [压缩probe](runs/texture_study/compression_probe/report.json) · [最新Token核查](runs/texture_study/band_tokens_comparison/token_similarity.json)

## 复现与权重

在earth-representation目录，配置现有数据发布和Python依赖后：

```powershell
python -m pytest tests/pd_here_geotoken_dualvit_r3 -q
python scripts/train_pd_here_r3_texture_study.py --help
python scripts/train_pd_here_r3_texture_study.py band_tokens256 --release outputs/pd_here_geotoken_dualvit_r3/texture_study --out outputs/pd_here_geotoken_dualvit_r3/texture_study/band_tokens256 --device cuda:0 --resume
```

训练参数和数据哈希见各组identity.json；实际CLI以--help为准。原生数据缓存及.pt权重按仓库规则留在本地；[checkpoint清单](checkpoint_inventory.json)登记路径、大小及SHA256。最新权重为outputs/pd_here_geotoken_dualvit_r3/texture_study/band_tokens256/last.pt。Viewer根目录为outputs/pd_here_geotoken_dualvit_r3/viewer，最新实验为texture30_band_tokens256_band_tokens256。静态预测图已随此归档发布。

原始设计包位于docs/PD_HERE_GeoToken_DualViT_R3，属于实施起点；最新架构以src/pd_here_geotoken_dualvit_r3/band_tokens.py与对应identity.json为准。

上传快照在独立目录验证：10项pytest测试通过，训练入口--help加载成功；既有GPU与恢复验证见对应运行JSON。上面的--resume命令依赖本地修正数据与last.pt，完整预算已完成时会保持完成状态；重新训练应换新的--out并去掉--resume。
