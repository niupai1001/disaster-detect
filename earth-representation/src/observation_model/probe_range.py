"""Read one official native COG window through the metered range adapter."""
import json
from pathlib import Path
from urllib.parse import urlsplit

import rasterio
from pyproj import Transformer

from .metered_http import DownloadBudget
from .native_data import native_window


def sign_asset(url,budget):
    parts=urlsplit(url);account=parts.hostname.split(".")[0];container=parts.path.split("/")[1]
    token=budget.json_request("GET",f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}")["token"]
    return url+"?"+token


def main():
    config=json.loads((Path(__file__).parent/"config.json").read_text())
    if not config.get("download_enabled",False):
        print("DOWNLOADS_PAUSED_BY_USER: no network probe",flush=True)
        return
    out=Path(__file__).parent/"outputs/phase0";budget=DownloadBudget(out/"download_budget.json",10737418240)
    candidate=json.loads((out/"probe_catalogue.json").read_text())["candidate"]
    b=candidate["bbox"];tr=Transformer.from_crs(candidate["crs"],4326,always_xy=True);ll=tr.transform(b[0],b[1]);ur=tr.transform(b[2],b[3])
    features=budget.json_request("POST","https://planetarycomputer.microsoft.com/api/stac/v1/search",json=dict(collections=["sentinel-2-l2a"],bbox=[*ll,*ur],datetime=candidate["s2_date"]+"T00:00:00Z/"+candidate["s2_date"]+"T23:59:59Z",limit=10))["features"]
    feature=features[0];url=sign_asset(feature["assets"]["B02"]["href"],budget)
    def opener(path,mode="r"):
        if str(path)!="band.tif":raise FileNotFoundError(path)
        return budget.open(url)
    bbox=[b[0]+40,b[1]+40,b[2]-40,b[3]-40]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        value,valid,transform,crs=native_window("band.tif",bbox,10,opener)
    print(json.dumps(dict(shape=value.shape,dtype=str(value.dtype),valid=float(valid.mean()),transform=list(transform)[:6],crs=crs,received_bytes=budget.received)))


if __name__=="__main__":main()
