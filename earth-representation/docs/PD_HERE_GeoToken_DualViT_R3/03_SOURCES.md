# 来源与适用边界

访问日期：2026-11-17。来源用于标准模块/MAE/运行接口，不作为R3有效性的实验证据。未检查最新用户仓库或本地原始数据。本地参考测试使用的PyTorch版本见05，不假设等于网页文档版本。

[S1] Dosovitskiy et al. An Image is Worth 16x16 Words，§3.1/Appendix A：直接线性Patch投影、位置编码、Transformer Block。
```text
https://arxiv.org/html/2010.11929v2
```
[S2] He et al. Masked Autoencoders Are Scalable Vision Learners，§3：可见Token Encoder、共享mask与完整Decoder、只重建hidden patch、预训练后丢弃Decoder。
```text
https://arxiv.org/html/2111.06377v3
```
[S3] 官方MAE模型实现：random_masking、forward_encoder、forward_decoder和forward_loss。
```text
https://github.com/facebookresearch/mae/blob/main/models_mae.py
```
[S4] PyTorch DDP官方文档：梯度同步、输入并非自动分片、no_sync与unused参数注意事项。网页版本不替代实际runtime锁定。
```text
https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
https://docs.pytorch.org/docs/stable/distributed.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
```
[S5] 用户附件《PD-HERE_V1_架构审阅与性能诊断报告.docx》，审查提交297e010…：§4要求冻结成员/QA/定标/目标和公平曝光；§6要求工程、可学习性、重建和表征分开验收。旧报告不覆盖后续R3新方案，旧倍率不作本轮目标。

方案自有定义：两个分支各压为一Token、特征级融合、固定S2槽位处理B1、几何Patch级单Token输出、7:3与batch16、预算数值和阶段预算。它们是当前设计和工程起点，不冒充上述论文结论。
