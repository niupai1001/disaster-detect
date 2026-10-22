# 历史实验归档：按需读取

默认先读 `../docs/RESEARCH_INDEX.md`，不批量阅读本目录。阶段目标和变化在固定模型文档，当前结果在两份固定数据文档。

`experiment_history/docs/` 保存移入的旧阶段文档与配图，`experiment_history/reports/` 保存原reports的稳定快照。原reports运行路径保留，避免影响其他进程写日志；占用或继续变化的记录在catalog中说明，不作为最终完成产物。训练目录、配置、权重、数据均未移动。

`experiment_history/catalog.json` 提供原路径、归档路径、SHA256与快照说明。归档原文不重写；旧文中的相对链接按原位置理解，查询工具会返回其实际目标。旧计划只代表历史，不是执行授权。

另已为105个登记run保存464份配置、训练摘要、元数据、曝光记录和曲线快照；权重、预测数组及原训练目录仍原地保留。当前目录共884项可校验记录，含原始用户计划及一个日志后续版本。4个日志在截取时被其他进程占用，已保留快照和原路径；它们在catalog的pending字段中标注，不当作最终日志。一个文档在归档期间被其他进程更新，catalog同时保留搬移前哈希与实际快照哈希，未掩盖此变化。

按需查询示例（项目根目录）：

```powershell
& 'C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe' -X utf8 scripts/query_research.py --history v4r4 --limit 5
& 'C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe' -X utf8 scripts/query_research.py --history-file docs/v6_temporal_spectral_2026-10-21.md --lines 40
& ./scripts/archive_research_history.ps1 -Verify
```

前一命令只返回文件元数据，第二条是显式限量读取并验证快照哈希；没有默认全文dump。`.rgignore` 排除历史与原run，确需全文检索时显式 `rg --no-ignore ... archive/experiment_history/docs`，限定路径。
