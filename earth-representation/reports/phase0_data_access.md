# Phase 0 Report — 数据可用性与项目骨架

生成时间：2026-10-31
项目根目录：`N:\Users\yzh\Disater\earth-representation`
执行依据：《物理引导的跨卫星通用地表表征学习：分阶段执行计划》Phase 0

## 1. 执行范围

完成了 Phase 0 的第 1–7 项与第 9 项，第 8 项（EnMAP）只完成目录盘点、下载未完成。

已做：
- 项目骨架与配置；复用 `landslide-seg\.venv-use-independent` 的 Python 3.12（rasterio 1.5.2 / GDAL 3.12.2），未新建环境、未新增依赖。
- HLS V2.0 官方目录（CMR-STAC，`HLSS30_2.0` / `HLSL30_2.0`）实际查询与统计。
- HLS 4 景（S30 两景、L30 两景）真实读取与物理量验证。
- Sentinel-2 L2A、Landsat 8/9 C2 L2 的目录查询与各一景读取测试。
- EnMAP L2A 目录盘点（408 景，218 景可用），下载准入探测。
- 数据量核算与失败记录。

未做（按要求停在 Phase 0）：
- 未实现任何模型、未训练。
- 未做 HLS 大规模下载（Phase 1 才批量取 patch）。
- EnMAP 未下载任何景（认证被拒）。

## 2. 数据

AOI 来源：现有工程的 `s2_10band_v4_p20.jsonl` 中每个 patch 的投影坐标 bbox，按事件取并集并转 WGS84，写入 `configs/regions.yaml`。共 15 个区域，范围 0.05°–1.94°。

| 区域 | HLS S30 | HLS L30 | S2 景 | S2 晴空 | L8/9 景 | L8/9 晴空 | EnMAP 景 | EnMAP 可用 |
|---|---|---|---|---|---|---|---|---|
| chimanimani | 100 | 82 | 293 | 129 | 44 | 14 | 2 | 1 |
| china | 76 | 44 | 18 | 0 | 106 | 7 | 4 | 4 |
| dominicamaria | 100 | 46 | 116 | 27 | 46 | 5 | 5 | 3 |
| hiroshima | 100 | 100 | 500 | 85 | 67 | 10 | 43 | 16 |
| hokkaido | 49 | 33 | 72 | 4 | 45 | 1 | 7 | 1 |
| indonesia | 77 | 89 | 16 | 0 | 102 | 6 | 3 | 0 |
| italy | 100 | 100 | 500 | 201 | 140 | 55 | 192 | 103 |
| itogon | 100 | 100 | 500 | 113 | 67 | 28 | 12 | 5 |
| kyrgyzstan1 | 174 | 200 | 86 | 33 | 88 | 35 | 46 | 38 |
| kyrgyzstan2 | 171 | 200 | 79 | 14 | 99 | 35 | 20 | 10 |
| nepal | 91 | 43 | 27 | 0 | 98 | 28 | 7 | 0 |
| newzealand | 200 | 200 | 289 | 68 | 162 | 19 | 30 | 21 |
| thrissur | 100 | 38 | 211 | 51 | 22 | 4 | 10 | 4 |
| usa_alaska | 199 | 81 | 179 | 18 | 101 | 17 | 6 | 4 |
| usa_puertorico | 100 | 44 | 142 | 33 | 44 | 6 | 21 | 8 |
| **合计** | 1737 | 1400 | 3028 | 776 | 1231 | 270 | 408 | 218 |

口径说明：
- HLS 计数是每个事件日期前后 180 天窗口内的 granule 数，单窗口上限 500，因此是“窗口内下限”，不是全存档总量。
- S2 与 Landsat 同样为 ±180 天窗口；S2 单窗口上限 500，hiroshima、italy、itogon 已达上限。
- “晴空”= 目录云量 ≤10%。EnMAP 的“可用”= 目录云量 ≤20%（EnMAP 景更大、云更敏感）。
- EnMAP 计数覆盖整个任务存档（2022-04-27 起），**不对应灾害事件日期**；15 个事件里只有 italy（2023-05-16）落在存档期内。

物理与质量口径（`configs/data_sources.yaml` 为唯一来源）：
- HLS：int16，nodata −9999，反射率 = DN × scale + offset，scale/offset 从产品 XML 读取（实测 0.0001 / 0）；Fmask 位 0–5 = fill/cloud/adjacent-cloud/shadow/snow/water。
- S2：uint16，按 SAFE 的 `BOA_QUANTIFICATION_VALUE` 与 `BOA_ADD_OFFSET` 反标定，不统一除 10000。
- Landsat：uint16，`DN × 0.0000275 − 0.2`，QA_PIXEL/QA_RADSAT 按 C2 位定义解码，只取 L8/L9。

## 3. 实现

| 文件 | 作用 |
|---|---|
| `scripts/build_regions.py` | 从现有 patch 清单生成 15 个 AOI，写 `configs/regions.yaml` |
| `scripts/_common.py` | 环境变量、Earthdata token、CMR-STAC/PC 查询、窗口读取、清单写入 |
| `scripts/audit_hls_stac.py` | HLS 目录统计 + 按 AOI 晴空率选景 + 窗口读取与验证 |
| `scripts/audit_s2_access.py` | S2 目录统计 + 一景 11 波段窗口读取 |
| `scripts/audit_landsat_access.py` | Landsat L8/9 目录统计 + 一景 7 SR + 2 QA 窗口读取 |
| `scripts/inventory_enmap.py` | EnMAP 目录盘点 + 下载准入探测 |

关键机制：**所有下载都是远程 COG 的 AOI 窗口读取**，网络传输与落盘都只包含 AOI 像素，不存在“下整景只用一角”。

运行命令：

```bash
python scripts/build_regions.py
python scripts/audit_hls_stac.py
python scripts/audit_s2_access.py
python scripts/audit_landsat_access.py
python scripts/inventory_enmap.py --probe
```

## 4. 验收

| 计划验收项 | 结果 | 证据 |
|---|---|---|
| HLS 两个 V2.0 collection 均可目录查询 | 通过 | `hls_availability.json`：S30 1737、L30 1400 |
| 至少各成功读取 2 个 S30/L30 granule 或有效子集 | 通过 | 4 景窗口读取，13.1 MB，`hls_availability.json.downloads` |
| HLS 反射率、QA、角度与 metadata 读取测试 | 通过 | CRS UTM、像元 30.0 m、nodata −9999、Fmask 33 个取值、SZA/SAA 已读、XML scale=0.0001 offset=0 |
| S2、Landsat 至少各一景读取测试 | 通过 | S2 11 波段 99.6 MB；Landsat 9 图层 5.5 MB |
| 所有失败项有日志与原因 | 通过 | 见第 6 节 |
| 不把“目录有”当成“AOI 够用” | 通过 | 选景改为按 AOI 晴空率，见第 6 节第 4 条 |
| EnMAP 只做少量下载 | **未完成** | 认证 401，见第 6 节第 1 条 |

## 5. 结果

- 下载总量：HLS 13.1 MB + S2 99.6 MB + Landsat 5.5 MB ≈ **118 MB**（窗口化后）。
- 4 景 HLS 的 AOI 晴空率：0.951 / 0.997 / 0.997 / 0.998。
- 若按整景下载同样 4 景 HLS 需 206 MB，窗口化节省约 94%。
- 目录侧结论：15 个区域在四个源上都有数据；瓶颈不是覆盖，是**时间对齐**（EnMAP）与**认证**（EnMAP、2013–2019 的 HLS 本可用但需要 Earthdata，本报告已解决）。

## 6. 失败与风险

1. **EnMAP 下载被拒（未完成）**。`WildDog` 与 `2260415@bjfu.edu.cn` 两个用户名在 `sso.eoc.dlr.de/eoc/auth/login` 均返回 HTTP 401，试过：原始表单提交、整页字段合并、浏览器等价字段序、邮箱作用户名。该服务同时把登录委派给 `identity.cloudferro.com/auth/realms/CODE-DE3`（Code-DE/EO-Lab 身份源）。目录侧 408 景可用，但文件一个都无法取回。**按用户要求记为未完成。**
2. **Earth Search 弃用**。S2 初版用 Earth Search，连续出现 HTTP 502（15 个区域中 8 个失败），改用 Planetary Computer 的同一 L2A 产品后 15/15 成功。这是对计划“CDSE 官方入口”的偏离，已在 `data_sources.yaml` 记录。
3. **GDAL 头注入方式**。`GDAL_HTTP_HEADERS` 在本机 GDAL 3.12.2 不生效，LP DAAC 的带鉴权 COG 打不开；改用 `GDAL_HTTP_HEADER_FILE` 后正常。此差异已写进 `_common.py` 注释，换机器需重测。
4. **选景逻辑曾有缺陷（已修）**。首版按目录云量最低选景，选中的 4 景里 3 景 AOI 填充率 74%–96%（云量显示 0 是因为大片无数据）。现改为先探测 AOI 的 Fmask，按“晴空率”排序，并要求 ≤10% 晴空率不入选；4 景晴空率全部 ≥0.95。**这条同时说明：只看云量或只看有效像元都不够，必须合并判断。**
5. **HLS 计数是窗口内下限**，不能当作全存档总量；Phase 1 若需要更多景，要放大窗口或分页。
6. **EnMAP 与事件时间不匹配**：存档从 2022-04-27 起，15 个事件仅 italy（2023）可用。若只为光谱多样性做补充实验可以，但不能与 2016–2019 的事件配对。

## 7. 下一阶段建议

Phase 0 的数据链已验证，可以进入 Phase 1。Phase 1 的计划与用户已确认的决定：

- 影像操作沿用现有工程：512×512 patch、10 m 参考网格、20 m 波段插值到 10 m 但保留原生 GSD 描述。
- split 按区域：train 11 区、val 2 区（hiroshima、italy）、test 2 区（newzealand、usa_puertorico）。
- EnMAP 适配器只写接口、不启用，准入结论写入 Phase 1 报告。
- 所有 patch 通过窗口读取直接从 COG 生成，不落地整景；预计 2.5 GB 以内。

需要用户决策（已在执行前提出）：EnMAP 账号问题的处理方式；patch 数量上限。
