# earth-representation

物理引导的跨卫星通用地表表征学习 — Phase 0/1 数据链。

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
