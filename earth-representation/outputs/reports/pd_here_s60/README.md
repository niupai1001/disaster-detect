# S60 结构实验

固定校正 S2 数据，700 TRAIN／300 开发 VAL。各组 seed 17、30 epoch、1,320 次更新，按 last 比较。

- [S60 Dual-Axis 对照](matched_seed17_v1/findings.md)
- [S60 空间 ViT 消融](spatial_vit_seed17/findings.md)
- [原生窗几何](windows.json)
- [检查点哈希](checkpoint_inventory.json)

结果目录保留逐轮指标、预测图、固定样本内容干预、运行身份及相关工程验证。训练依赖项目本地的修正数据发布，权重按仓库规则保存在本地。
