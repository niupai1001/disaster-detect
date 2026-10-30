"""Real released optical subsets and missing-modality paths; zero updates."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
from landslide_seg.data.use_integrated import IntegratedUSEData
from landslide_seg.models.use_next_v2 import load_asset
from landslide_seg.models.use_next_v2.physical import area_grid
from landslide_seg.models.use_next_v2.pipeline import USEModel
from device import claim


def main():
    torch.set_num_threads(4);device='cuda:0'
    with claim(device) as uuid:
        model=USEModel(ROOT).to(device).eval();report=[]
        for sensor,count in (('s2',3),('s2',4),('s2',6),('s2',10),('s2',12),('landsat',6),('landsat',7),('s1',2)):
            dataset=IntegratedUSEData(ROOT/'outputs/use_next_v2/data/data_release_v1',sensor=sensor)
            index=next((i for i,r in enumerate(dataset.rows) if len(r['band_ids'])>=count),None)
            if index is None:
                report.append(dict(sensor=sensor,channels=count,status='NOT_AVAILABLE_REAL_SUBSET'));continue
            a=load_asset(dataset,index)
            for key in ('image','pixel_valid','channel_metadata','metadata_known'):setattr(a,key,getattr(a,key)[:count].to(device))
            a.channel_names=a.channel_names[:count]
            _,shape,t,crs=area_grid(a)
            with torch.no_grad():out=model(a,dict(crs=crs,shape=shape,origin=(t[2],t[5])))
            assert torch.isfinite(out.global_embedding).all()
            report.append(dict(sensor=sensor,channels=count,status='PASS',real_asset=a.provenance['id'],semantic_paths=list(out.semantic_levels)))
        (ROOT/'outputs/use_next_v2/engineering/modalities.json').write_text(json.dumps(dict(uuid=uuid,rows=report,formal_updates=0),indent=2))
        print(json.dumps(report))


if __name__=='__main__':main()
