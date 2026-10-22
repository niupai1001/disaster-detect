import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely import wkt as shapely_wkt
import numpy as np

tif_path = "database/第一批数据/1.4火灾LS/GEE_LSHUO_20260101_172755/T_C5_49RFH_47_EV20191011_IM20190918_F13.tif"
csv_path = "/Users/Lemon/Documents/New project/database/csv文本/泥石流火灾.csv"

# ---------- 读取影像 ----------
with rasterio.open(tif_path) as src:
    img = src.read(1)
    crs = src.crs
    bounds = src.bounds           # 地理范围 (left, bottom, right, top)
    transform = src.transform
    print("影像 CRS:", crs)
    print("影像范围:", bounds)

    # 拉伸
    vmin, vmax = np.percentile(img[img > 0], [2, 98])
    img_disp = np.clip(img, vmin, vmax)
    img_disp = (img_disp - vmin) / (vmax - vmin)

# ---------- 读取 mask ----------
df = pd.read_csv(csv_path)
geom_list = df['geometry_wkt'].apply(shapely_wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry=geom_list, crs="EPSG:4326")
print("原始 mask CRS:", gdf.crs)
print("mask 总范围:", gdf.total_bounds)   # [minx, miny, maxx, maxy]

# ---------- 转换投影（如果需要） ----------
if crs != gdf.crs:
    print(f"投影不一致，正在转换 mask 到 {crs} ...")
    gdf = gdf.to_crs(crs)
    print("转换后 mask 范围:", gdf.total_bounds)
else:
    print("投影已一致，无需转换。")

# ---------- 对齐绘制 ----------
fig, ax = plt.subplots(figsize=(12, 12))

# 用 rasterio.plot.show 绘制影像，自动配准地理坐标
ax = show(img_disp, transform=transform, ax=ax, cmap='gray', vmin=0, vmax=1)

# 绘制 mask
gdf.plot(ax=ax, column='category', cmap='Set1',
         alpha=0.6, edgecolor='black', linewidth=2,
         legend=True, legend_kwds={'label': 'Category'})

# 强制设置绘图范围为影像范围，避免 mask 飞出视野
ax.set_xlim([bounds.left, bounds.right])
ax.set_ylim([bounds.bottom, bounds.top])

ax.set_title("Satellite image with masks")
plt.tight_layout()
plt.show()