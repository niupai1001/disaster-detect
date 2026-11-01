# earth-representation

物理引导的跨卫星地表表征学习。当前入口为[修复、评价与接续报告](reports/verification.md)，实际状态见[工作流](outputs/verification/workflow_state.json)。

本轮完成9个单种子自监督实验、19个同头滑坡冻结诊断、18个WorldCover产品一致性诊断、基线与最佳新候选微调，以及HLS/DOFA比较。配对目标改善跨卫星检索，尚未同时改善辅助语义任务。独立人工土地覆盖标签仍缺失，EnMAP不阻塞主线。

## 本轮复现与交付

训练日志和固定案例：[Training Viewer](http://127.0.0.1:8766/)，TVLP 1.0；[配置](configs/verified_workflow.json)、[实际预算与权重哈希](outputs/verification/training_acceptance.json)、[编码器导出说明](outputs/verification/export/README.md)。历史源数据和权重不覆盖，恢复早期三个单源运行须用acceptance中指向的`resume_fixed`副本。

在本项目根目录的PowerShell中运行（重训需新tag和独立输出，不覆盖本轮实验）：

```powershell
$py='N:\Users\yzh\Disater\landslide-seg\.venv-use-independent\Scripts\python.exe'
$env:EARTH_S2_ROOT='N:\Users\yzh\Disater\landslide-seg\data\sen12'
$env:EARTH_PROTOCOL_MANIFEST='N:\Users\yzh\Disater\landslide-seg\manifests\v4r1\pool_100.jsonl'
$env:CUDA_VISIBLE_DEVICES='1'
& $py -m scripts.setup_dino
& $py -m scripts.launch_dino --device cuda --adaptive --source landslide --n-patches 2223 --batch-size 32 --seed 0 --steps 50000 --max-epochs 400 --tag reproduced_s2 --out outputs/reproduced/runs --tv-root outputs/reproduced/tvlp
# 两源匹配队列：使用同一冻结290位置；route可取single、mixed、paired。
& $py -m scripts.launch_dino --device cuda --adaptive --route paired --sensor-release outputs/verification/sensor_release --batch-size 32 --seed 0 --steps 50000 --max-epochs 400 --tag reproduced_paired --out outputs/reproduced/runs --tv-root outputs/reproduced/tvlp
# 以下命令复评/重建报告，已有任务结果会被复用。
& $py -m scripts.evaluate_geographic_retrieval --include-routes
& $py -m scripts.accept_verified_runs
& $py -m scripts.render_verification
& $py -m scripts.package_verified
```

完整原运行参数和成员在`outputs/verification/runs/*.provenance.json`；精确旧源码在`release`、`augmentation_release`、`scale_release`、`sensor_code_release`四个冻结目录，其manifest含哈希。当前入口额外修复轮次边界恢复记录、失败生命周期和资源日志，不能把新入口哈希冒充旧运行源码。各队列脚本记录原执行命令，原失败、恢复命令和返回值也保留。新运行应先向Viewer attach日志目录，命令见下面说明。

```powershell
$viewer='C:\Users\yzh\Downloads\training_viewer_v1.1_spec_and_examples\training_viewer_v1\.venv\Scripts\training-viewer.exe'
& $viewer attach 'N:\Users\yzh\Disater\earth-representation\outputs\reproduced\tvlp' --data-dir 'N:\Users\yzh\Disater\earth-representation\outputs\verification\viewer'
& $viewer validate 'N:\Users\yzh\Disater\earth-representation\outputs\reproduced\tvlp\reproduced_s2' --json
```

训练使用FP32，未采用BF16；本轮只运行已有测试与关键路径检查。下列Phase 0/1命令保留作历史数据入口，不表示当前自动下载或继续EnMAP。

## 环境

复用 `../landslide-seg/.venv-use-independent` 的 Python 3.12（rasterio 1.5.2 / GDAL 3.12.2）。
未新建环境、未新增依赖。

## 凭证

`.env`（已被 `.gitignore` 排除）：

```
EARTHDATA_USER=...
EARTHDATA_PASS=...
ENMAP_USER=...
ENMAP_PASS=...
```

## 运行

```bash
python scripts/build_regions.py          # 15 个事件 AOI -> configs/regions.yaml
python scripts/audit_hls_stac.py         # HLS 目录 + AOI 窗口读取验证
python scripts/audit_s2_access.py        # S2 目录 + 一景窗口读取
python scripts/audit_landsat_access.py   # Landsat 目录 + 一景窗口读取
python scripts/inventory_enmap.py --probe
python scripts/build_patches.py          # Phase 1 patch 清单
python -m pytest tests -q                # Phase 1 测试
```

## 关键约束

- 所有数据都是**远程 COG 的 AOI 窗口读取**，从不下载整景。
- GDAL 带鉴权读取必须用 `GDAL_HTTP_HEADER_FILE`（`GDAL_HTTP_HEADERS` 在本机无效）。
- Landsat 只取 landsat-8/9；EnMAP 下载在 Phase 0 被 401 挡住，适配器仅保留接口。

## 目录

- `configs/` 区域定义与数据源契约
- `scripts/` 审计与构建脚本
- `src/data/` 波段目录、QA、适配器、patch 与清单
- `data/manifests/` 四个可用性清单 + train/val/test/provenance
- `reports/` Phase 0 / Phase 1 报告

独立训练队列的原始状态与最终验收可能不同（原生失败后另行恢复）；最终以`workflow_state.json`和`training_acceptance.json`核对，原队列状态不涂改为成功。
