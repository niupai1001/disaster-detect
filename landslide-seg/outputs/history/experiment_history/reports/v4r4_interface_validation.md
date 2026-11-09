# v4r4 接口验证报告（训练前门槛）

结论：**全部通过**

设备 `cuda:0`；固定样本 8 个。

## 1. 初始参数身份

- 公共参数张量 218 个；新增交换参数 12 个
- LateDetail 公共哈希 `c9deeddce71f0766bbeb36a9674f3db5`
- Self 公共哈希 `c9deeddce71f0766bbeb36a9674f3db5`
- Cross 公共哈希 `c9deeddce71f0766bbeb36a9674f3db5`
- Self 新增哈希 `016ff85f27da7ebdf02f27e7f8c188ae`
- Cross 新增哈希 `016ff85f27da7ebdf02f27e7f8c188ae`
- 交换参数键：exchange.alpha_d, exchange.alpha_e, exchange.h_d.0.bias, exchange.h_d.0.weight, exchange.h_d.2.bias, exchange.h_d.2.weight, exchange.h_e.0.bias, exchange.h_e.0.weight, exchange.h_e.2.bias, exchange.h_e.2.weight, exchange.p.bias, exchange.p.weight

## 2. 门槛逐项

| 门槛 | 检查项 | 结果 | 说明 |
|---|---|---|---|
| setup | 固定 8 个 TRAIN 样本（无增强） | PASS | n=8, events=['chimanimani', 'dominicamaria', 'hokkaido', 'itogon', 'kyrgyzstan1', 'usa_puertorico'] |
| G2 | 公共参数初始哈希：LateDetail vs Self | PASS | c9deeddce71f0766 |
| G2 | 公共参数初始哈希：Self vs Cross | PASS | c9deeddce71f0766 |
| G2 | 新增模块初始哈希：Self vs Cross | PASS | 016ff85f27da7ebd |
| G2 | Self 相对 LateDetail 只多出 exchange.* | PASS | extra=12, missing=0 |
| G1 | 输入/阶段2/E校正/D/阶段3/阶段4 形状符合计划 | PASS | input=[2, 10, 128, 128], stage1=[2, 32, 32, 32], E_in=[2, 64, 16, 16], E_out=[2, 64, 16, 16], D_in=[2, 32, 128, 128], D_out=[2, 32, 128, 128] |
| G1 | E_new 确实进入阶段 3（改 α_E 后 stage3 变化） | PASS | max_abs_delta_stage3=0.2735706865787506 |
| G2 | eval 下零系数初始输出 vs LateDetail（self） | PASS | max_abs_diff=0.0 |
| G2 | eval 下零系数初始输出 vs LateDetail（cross） | PASS | max_abs_diff=0.0 |
| G3 | 参数分组无重复、无遗漏（三架构） | PASS | late={'dup': False, 'missing': 0}, self={'dup': False, 'missing': 0}, cross={'dup': False, 'missing': 0} |
| G3 | Self / Cross 可训练参数精确相等 | PASS | self={'total': 3777442, 'trainable': 3777442, 'encoder': 3330368, 'decoder': 395009, 'new': 52065}, cross={'total': 3777442, 'trainable': 3777442, 'encoder': 3330368, 'decoder': 39 |
| G3 | 交换模块参数落在 new 组（6e-4），未被并入 encoder | PASS | late=True, self=True, cross=True |
| G3 | 学习率组只有 6e-5(encoder) 与 6e-4(其余) | PASS | 6e-05, 0.0006 |
| G4 | 20 步内两方向均获得有限梯度（self） | PASS | p=19, h_e=19, h_d=18, alpha_e=20, alpha_d=19 |
| G4 | 20 步内 loss 有限（self） | PASS | loss_last=0.5682408213615417 |
| G4 | 20 步内两方向均获得有限梯度（cross） | PASS | p=18, h_e=19, h_d=18, alpha_e=20, alpha_d=19 |
| G4 | 20 步内 loss 有限（cross） | PASS | loss_last=0.5692792534828186 |
| G5 | α_D=0 时 D_new ≡ D（两架构） | PASS | cross=0.0, self=0.0 |
| G5 | α_D≠0 时 D_new 改变（两架构） | PASS | cross=0.8972458243370056, self=1.040486216545105 |
| G5 | 缩放 D 改变 Cross 阶段3 输入 | PASS | 0.6177685260772705 |
| G5 | 缩放 D 不改变 Self 阶段3 输入（无跨路影响） | PASS | 0.0 |
| G6 | 100 步后能在固定 8 样本上降 loss（self） | PASS | first=0.800564, last=0.336755 |
| G6 | 两方向残差系数已离开零（self） | PASS | alpha_e_max=0.02419967, alpha_d_max=0.03968981 |
| G6 | 100 步后能在固定 8 样本上降 loss（cross） | PASS | first=0.800564, last=0.338902 |
| G6 | 两方向残差系数已离开零（cross） | PASS | alpha_e_max=0.01296499, alpha_d_max=0.03856032 |
| G7 | checkpoint 保存/重载输出一致 | PASS | max_abs_diff=0.0 |
| G7 | batch64 前反向无 OOM 且梯度全有限 | PASS | batch=64, loss=0.834241, peak_alloc_mib=5506.1, peak_reserved_mib=6974.0, all_grads_finite=True, n_grad_tensors=227 |
| G7 | 全参数 FP32（无静默 AMP） | PASS | torch.float32 |

## 3. 参数量与分组

| 架构 | 总参数 | encoder | decoder | new | 交换模块 |
|---|---:|---:|---:|---:|---:|
| late | 3,753,602 | 3,330,368 | 395,009 | 28,225 | 0 |
| self | 3,777,442 | 3,330,368 | 395,009 | 52,065 | 23,840 |
| cross | 3,777,442 | 3,330,368 | 395,009 | 52,065 | 23,840 |

优化器学习率组：[{'lr': 6e-05, 'n_tensors': 192}, {'lr': 0.0006, 'n_tensors': 35}]

## 4. 100 步拟合（固定 8 样本，无增强）

| 架构 | 步 | loss | batch IoU | α_E max | α_D max |
|---|---:|---:|---:|---:|---:|
| self | 1 | 0.800564 | 0.0134 | 1.200e-04 | 0.000e+00 |
| self | 20 | 0.568778 | 0.0000 | 8.352e-03 | 9.608e-03 |
| self | 40 | 0.457275 | 0.0000 | 1.253e-02 | 2.114e-02 |
| self | 60 | 0.410047 | 0.3418 | 1.815e-02 | 3.098e-02 |
| self | 80 | 0.358008 | 0.4410 | 2.273e-02 | 3.746e-02 |
| self | 100 | 0.336755 | 0.4858 | 2.420e-02 | 3.969e-02 |
| cross | 1 | 0.800564 | 0.0134 | 1.200e-04 | 0.000e+00 |
| cross | 20 | 0.568759 | 0.0000 | 5.174e-03 | 9.755e-03 |
| cross | 40 | 0.457664 | 0.0000 | 7.203e-03 | 2.112e-02 |
| cross | 60 | 0.411993 | 0.3097 | 1.098e-02 | 3.004e-02 |
| cross | 80 | 0.360251 | 0.4162 | 1.248e-02 | 3.614e-02 |
| cross | 100 | 0.338902 | 0.4765 | 1.296e-02 | 3.856e-02 |

## 5. 显存与数值

```
{
 "batch": 64,
 "loss": 0.834241,
 "peak_alloc_mib": 5506.1,
 "peak_reserved_mib": 6974.0,
 "all_grads_finite": true,
 "n_grad_tensors": 227,
 "param_dtypes": [
  "torch.float32"
 ]
}
```

## 6. 干预诊断

| 架构 | α_D=0 时 D 变化 | α_D≠0 时 D 变化 | 缩放 D 后 stage3 变化 |
|---|---:|---:|---:|
| cross | 0.000e+00 | 8.972e-01 | 6.178e-01 |
| self | 0.000e+00 | 1.040e+00 | 0.000e+00 |
