# Disaster Research

两个独立研究子项目，共用必要的本机运行环境和历史资源。

| 子项目 | 研究范围 | 工程入口 | 设计入口 |
|---|---|---|---|
| landslide-seg | SegFormer、U-Net、UTAE滑坡分割 | [README](landslide-seg/README.md) | [DESIGN](landslide-seg/docs/DESIGN.md) |
| earth-representation | USE与后续跨卫星地表表征 | [README](earth-representation/README.md) | [DESIGN](earth-representation/docs/DESIGN.md) |

各项目采用src、configs、scripts、tests、data、weights、outputs、docs的固定分工；滑坡项目另有成员清单目录manifests。四份专题在docs，README只解释工程使用。原始数据、派生缓存、模型权重与运行记录不随源码重复存放。

根AGENTS规定维护规则；隐藏目录.project保存整理记录、兼容映射与维护工具。旧模块、资源和历史路径有必要的兼容映射，实际位置以各工程README为准。
