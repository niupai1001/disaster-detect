# V3-FR 本机复现

使用冻结发布的 `runtime`、`config.json` 与成员绑定。训练运行身份在各组 `identity.json`，每个检查点的位置和原始字节哈希列在 [产物清单](artifact_manifest.json)。查看历史复算时使用历史 Run 的 `best.pt`。新试验的 233 步端点使用该组 `update_233.pt`。旧架构的精确执行源码保存在 [B0/BP 冻结工作副本](../source_revision_b0_bp/)，后四组见 [N0–C 冻结工作副本](../source_revision_n0_through_c/)。

运行目录为 `earth-representation`。使用本机项目的 Python 3.12 环境，具体版本见 [环境记录](../environment.json)。`SOURCE` 为独立 V3-FR 源码目录，`CAMPAIGN` 为本产物目录，`GALLERY` 为其中的 `visualizations` 目录。以下命令读取原始预测、已保存评价与 [架构名称](../experiment_display_names.json) 来重绘图集。

```text
python -B SOURCE/scripts/plot_evidence.py CAMPAIGN GALLERY/figures
python -B SOURCE/scripts/spatial_screen.py CAMPAIGN --figures-only GALLERY/figures
python -B SOURCE/analysis/plot_endpoints.py CAMPAIGN GALLERY/figures
python -B SOURCE/analysis/plot_all_runs.py CAMPAIGN GALLERY
```

在本机打开 Training Viewer 的「Encoder V3-FR · 模块对照」和「Encoder V3-FR · 全可见拟合」。[Viewer 索引](../viewer_index.json) · [Viewer 与原始日志核对](../comparison_audit.json)
