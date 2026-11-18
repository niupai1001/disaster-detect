# S60 结构实验

固定校正 S2 数据，700 TRAIN／300 开发 VAL。各组 seed 17、30 epoch、1,320 次更新，按 last 比较。

- [S60 Dual-Axis 对照](matched_seed17_v1/findings.md)
- [S60 空间 ViT 消融](spatial_vit_seed17/findings.md)
- [原生窗几何](windows.json)
- [检查点哈希](checkpoint_inventory.json)

结果目录保留逐轮指标、预测图、固定样本内容干预、运行身份及相关工程验证。训练依赖项目本地的修正数据发布，权重按仓库规则保存在本地。

## 2,000景整景对照

两组各训练10轮、1,250步。复用同一12来源训练子集，保持原生栅格和512×512输出上限。固定22景TRAIN面板的A/B1隐藏像元完全一致，结果反映训练样本内表现。

| 配置 | 参数量 | 每轮耗时中位数 | 10轮累计 |
|---|---:|---:|---:|
| 60米空间ViT | 44,647,538 | 46.3秒 | 518.5秒 |
| 120米旧双分支 | 122,963,696 | 215.7秒 | 2138.3秒 |

旧双分支的参数量为2.75倍，轮耗时为4.66倍。参数量增加是精度改善的一个待验证解释。地理Patch大小、分支、深度和大景切分也发生了变化，需参数量匹配的受控实验才能拆分影响。

- [60米空间ViT10轮报告](native512_expanded2000_seed17/findings.md)
- [旧双分支10轮对照](native512_legacy_dual2000_v2_seed17/findings.md)
- [A预测同目标对照](native512_legacy_dual2000_v2_seed17/A_predictions_comparison.png)
- [两组末轮检查点哈希](whole_scene_checkpoint_inventory.json)
- [冻结成员](subset_expanded_2000_seed17.json)

训练依赖本地冻结核心和扩展发布，目录路径及哈希见运行记录。检查点与原始缓存保存在本地。
