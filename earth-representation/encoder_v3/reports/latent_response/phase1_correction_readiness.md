# Encoder V3当前训练前就绪状态

S2/Landsat E1：PASS。用户已确认IR1/IR5/IR6，完整LR模型与合成链路已实现；309维联合读出、共享Query投影、统一场景参考生效。真实更新0；E2/E3未授权。

EnMAP：BLOCKED_NOT_ADMITTED。两景原生影像缺配套QA，其中一景缺XML；两景产品B131–B135整景NoData。1689个派生窗口均已重采样且缺完整原生合同，未准入。

当前结果、准确测试计数、范围限制与全部证据见[执行报告](plan_execution/execution_report.md)、[E1](plan_execution/e1/readiness.json)、[EnMAP准入](plan_execution/enmap/admission.json)。旧phase1报告及纠错测试保留为历史证据，不能代替本次E1。

最大2769×2264完整十波段缓存仅核验Encoder前向，未证明完整Decoder/反向显存上界。CPU float64恢复下一更新一致；未宣称GPU恢复逐位一致。
