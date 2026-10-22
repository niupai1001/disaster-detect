# B2 接口验证（训练前门槛）

结论：**全部通过**

## 1. 权重身份

- 目录 `N:/Users/yzh/Disater/landslide-seg/weights/segformer/segformer-b2-finetuned-ade-512-512`
- `pytorch_model.bin` 110,000,905 B，sha256 `187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3`
- config: decoder 768，depths [3, 4, 6, 3]，hidden [64, 128, 320, 512]
- 张量标记: linear_c→768，classifier [150, 768, 1, 1]，首层 [64, 3, 7, 7]
- 下载元信息: `{"repo": "nvidia/segformer-b2-finetuned-ade-512-512", "revision": "de01bae28967510f9ddd496c60a969357195400c", "mirror": "https://hf-mirror.com", "downloaded_on": "2026-10-20", "files": {"config.json": {"bytes": 6885, "sha256": "ee7400840fdb1e5045f0b2eba78bf053df8e33a309c4acec31705a48c8cc5c00"}, "preprocessor_config.json": {"bytes": 271, "sha256": "8039d1d210abaa7117ad78e58cdfd6141a2ec72c03dae891b3cd76737e422c6c"}, "pytorch_model.bin": {"bytes": 110000905, "sha256": "187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3"}}, "config": {"model_type": "segformer", "num_labels": null, "hidden_sizes": [64, 128, 320, 512], "decoder_hidden_size": 768, "depths": [3, 4, 6, 3]}}`

> **命名不一致**：目录与 `download_meta.repo` 写的是 `…ade-512-512`，但 config 与全部张量都是 B2-768（decoder 768 / depths 3,4,6,3）。本报告以文件内容为准，未取得上游哈希校验（HF API 不可达），身份链只到本地文件。


## 2. 规格

```
{
 "backbone": "b2",
 "decoder_hidden": 768,
 "params": {
  "total": 27430338,
  "trainable": 27430338,
  "encoder": 24218240,
  "decoder": 3151105,
  "new": 60993
 },
 "decoder_feat_shape": [
  2,
  768,
  32,
  32
 ],
 "logits_shape": [
  2,
  1,
  128,
  128
 ],
 "delta_vs_b0_total": 23676736
}
```

## 3. 门槛

| 门槛 | 检查 | 结果 | 说明 |
|---|---|---|---|
| G1 | 权重文件内容 = B2-768 架构 | PASS | linear_c_out=768, classifier=[150, 768, 1, 1], first_conv=[64, 3, 7, 7] |
| G1 | 目录名/仓库名与内容一致（信息项，不阻断） | WARN | dir=segformer-b2-finetuned-ade-512-512, meta_repo=nvidia/segformer-b2-finetuned-ade-512-512, note=目录名写 512-512，但 config 与张量均为 B2-768；以内容为准，已在报告中标注 |
| G2 | from_pretrained 无 missing/unexpected/mismatched | PASS | missing_keys=0, unexpected_keys=0, mismatched_keys=0, error_msgs=0 |
| G3 | decoder 维度 = 768 且 logits 为 [B,1,128,128] | PASS | backbone=b2, decoder_hidden=768, params={'total': 27430338, 'trainable': 27430338, 'encoder': 24218240, 'decoder': 3151105, 'new': 60993}, decoder_fea |
| G3 | 参数量与离线核算一致（27,430,338） | PASS | 27430338 |
| G4 | batch64 前反向无 OOM 且梯度全有限 | PASS | batch=64, loss=0.822112, n_grad_tensors=387, all_finite=True, peak_alloc_mib=11421.4, peak_reserved_mib=15114.0 |
| G4 | 全参数 FP32 | PASS | ['torch.float32'] |
| G5 | 固定 8 样本 100 步能降 loss | PASS | first=0.820241, last=0.242902, iou_last=0.6024 |
| G6 | checkpoint 重载输出一致 | PASS | max_abs_diff=0.0 |
| G6 | 重载后参数哈希一致 | PASS | 637e1cc5351d6ae7 |
