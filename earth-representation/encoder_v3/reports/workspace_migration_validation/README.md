# 工作目录迁移验收

唯一工作目录为 `N:\Users\yzh\Disater\earth-representation`；Encoder V3 位于其 `encoder_v3` 子目录。

8061 个原文件共 3748302091 字节均已复制并通过 SHA256 校验。新目录运行59项工程测试全部通过，预检入口检查通过。两种优化器第100次完整检查点、源码身份、208次已用预算及下一次采样游标均保留，迁移新增真实参数更新为0。

C盘原目录改为兼容指向N盘的目录，原副本保存在同级 `encoder_v3_before_migration_backup`。旧报告与检查点里的绝对路径不批量改写。正式训练仍未启动。详细文件清单见上级 `workspace_migration_manifest.json`，位置与状态见上级 `workspace_relocation.json`。
