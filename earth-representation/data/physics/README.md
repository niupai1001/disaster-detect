# 传感器物理描述文件（供 Agent 读取）

本目录把所有可获取的传感器物理信息集中成两个入口文件。设计原则：**只写有来源的东西，缺什么就明确标缺**，不要用猜测值填空。

## 入口

| 文件 | 内容 |
|---|---|
| `sensor_physics.json` | 每个传感器的波段表、中心波长、FWHM、原生 GSD、辐射标定、SRF 曲线指针、PSF 状态 |
| `srf/*.csv` | SRF 曲线本体：第一列 `wavelength_nm`，其余每列一个波段 |
| `../datasets.json` | 本地各数据集的清单（路径、文件数、形状、波段、GSD、与 physics 的对应关系） |

## `sensor_physics.json` 结构

```jsonc
{
  "schema_version": "1.0",
  "sensors": {
    "S2A": {
      "instrument": "MSI",
      "bands": [
        {"band": "B4", "centre_nm": 664.62, "fwhm_nm": 29.0, "native_gsd_m": 10}
      ],
      "srf": {
        "csv": "srf/s2a_srf.csv",
        "status": "measured",           // measured | derived_gaussian | not_sourced
        "source": "sentinel2_msi_srf_v4_2024.xlsx",
        "grid_nm": [430.0, 960.0]
      },
      "radiometry": {"product": "L2A BOA reflectance", "scale": 0.0001,
                      "offset": 0, "nodata": 0},
      "psf": {"status": "not sourced"}
    }
  }
}
```

### status 的含义

| 值 | 含义 | 能不能用于 V3 训练 |
|---|---|---|
| `measured` | 来自官方实测文件（S2 的 COPE-GSEG-EIT 工作簿、Landsat 的 EUMETSAT rtcoef 表） | 可以 |
| `derived_gaussian` | 由产品元数据里的中心波长 + FWHM 生成的高斯曲线 | 只能当近似，必须在训练配置里标注 |
| `not_sourced` | 没拿到，留空 | 不能假装有 |

## 当前各传感器的 SRF 状态

| 传感器 | 波段数 | SRF | 来源 |
|---|---|---|---|
| S2A / S2B / S2C | 13 | `measured` | `srf/sentinel2_msi_srf_v4_2024.xlsx`（三张响应表） |
| Landsat 8 OLI | 9 | `measured` | `srf/landsat8_oli_srf_eumetsat.tar.gz`（Ball BA RSR v1.1，波数→波长已换算） |
| Landsat 9 OLI-2 | 9 | `measured` | `srf/landsat9_oli2_srf_eumetsat.tar.gz` |
| EnMAP HSI | 224 | `derived_gaussian` | 产品元数据 `wavelengthCenterOfBand` + `FWHMOfBand` |

EnMAP 的**官方实测 SRF 曲线**目前没拿到：DLR/GFZ 的 EnPT 包里也只是"用中心波长+FWHM 生成高斯"，没有附带实测表。所以这里标注为 derived，不作为实测值使用。

## PSF 状态

三个传感器都还没有可引用的 PSF 参数，因此统一写 `not_sourced`，**没有填任何猜测值**。如果要补，需要拿到：

- Sentinel-2：ESA 的 MSI PSF/MTF 文档（产品规范里的 PSF 章节）；
- Landsat 8/9：USGS Cal/Val 的 OLI 点扩散函数表征；
- EnMAP：DLR 的 HSI 相机 PSF 规格。

拿到后写进 `data/physics/srf/` 并在 `sensor_physics.json` 里把 `psf.status` 改成 `measured`，附来源文件名。

## 重建方式

```powershell
python -m scripts.build_sensor_physics    # 重新生成 sensor_physics.json 与 SRF CSV
python -m scripts.build_data_inventory    # 重新生成 datasets.json
```

两个脚本都只读本地文件，不联网。
