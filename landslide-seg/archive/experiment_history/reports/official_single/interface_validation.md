# BENCH_OFFICIAL 单期协议 · 最小验证

结论：**全部通过**

- 样本数：{'train': 3012, 'val': 758, 'test': 945}
- 协议：{"path": "N:\\Users\\yzh\\Disater\\landslide-seg\\manifests\\official_single\\protocol.json", "sha256": "16bbd71d1b8dfb9da959b9155e87844003daffdf2d1aa6d287ae385025b01044", "n_fit": 3012, "n_train": 3012, "n_dev": 758}
- 参数：{'total': 3753602, 'trainable': 3753602, 'encoder': 3330368, 'decoder': 395009, 'new': 28225}
- 预训练权重 sha256：`0f4df97633cbedd558ecffa3ad228ace5af37e082678390b45a9d22745787c61`

## 门槛

| 检查 | 结果 | 说明 |
|---|---|---|
| validate_protocol 通过 | PASS | path=N:\Users\yzh\Disater\landslide-seg\manifests\official_single\protocol.json, sha256=16bbd71d1b8dfb9da959b9155e87844003daffdf2d1aa6d287ae385025b01044, n_fit= |
| 三个 split 均非空 | PASS | train=3012, val=758, test=945 |
| train 读取正常 | PASS | [3, 10, 128, 128] |
| val 读取正常 | PASS | [3, 10, 128, 128] |
| test 读取正常 | PASS | [3, 10, 128, 128] |
| pooled IoU 与手算一致（全像素 1/3） | PASS | 0.3333333333333333 |
| valid 口径差异可复现（排除 FP 像素后 1/3 → 1/2） | PASS | all=0.3333333333333333, valid_only=0.5, note=官方 torchmetrics 用全像素；本项目主口径用 valid 像素。IoU/Dice/P/R 只在被排除像素为 FP 或 FN 时才变化，FPR 则必然变化。并表前须按同一口径复算 |
| 预训练权重已加载（非随机初始化） | PASS | missing=0, unexpected=0, pretrained=nvidia/segformer-b0-finetuned-ade-512-512 |
| batch64 前反向无 OOM 且梯度有限 | PASS | batch=64, loss=0.799288, n_grad=215, all_finite=True, peak_alloc_mib=4648.6 |
| 全参数 FP32 | PASS | ['torch.float32'] |
| checkpoint 重载输出一致 | PASS | max_abs_diff=0.0 |
