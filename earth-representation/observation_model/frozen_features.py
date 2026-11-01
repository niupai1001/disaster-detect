"""Frozen encoders on a common1200m auxiliary window; no label-dependent encoder selection."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import ROOT, RELEASE
from .model import BandRaster, make_model, checkpoint_coordinate_mode
from .native_data import S2

NAMES=["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]


def model_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def auxiliary_data(dataset):
    if dataset=="worldcover":
        folder=ROOT/"outputs/verification/worldcover_data"
        path=folder/"manifest.json";manifest=json.loads(path.read_text());rows=manifest["rows"]
        with np.load(folder/"arrays.npz") as z:
            h,w=z["image"].shape[-2:];yy=(h-60)//2;xx=(w-60)//2
            images=z["image"][:,:,yy:yy+60,xx:xx+60].copy()
            labels=z["label"][:,yy:yy+60,xx:xx+60].copy()
            valid=z["valid"][:,yy:yy+60,xx:xx+60].copy()
        gsd=20
        provenance=dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_array_sha256=manifest["arrays_sha256"],
                        source_role="existing WorldCover weak map agreement only",crop_source_pixels=[xx,yy,60,60],
                        boundary="20m cached inputs; 10m bands are interpolated inputs for this auxiliary diagnostic, not original native10m observations; no10m detail truth")
    else:
        path=ROOT.parent/"landslide-seg/manifests/v4r1/pool_100.jsonl"
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        images=[];labels=[];valid=[]
        for row in rows:
            if row["bands"]!=NAMES:raise ValueError("auxiliary cache band order differs")
            with np.load(row["cache"]) as z:
                v=z["valid"].astype(bool).T&(z["mask"].T!=255)
                image=z["x10"].transpose(0,2,1).astype(np.float32)/10000
                images.append(image[:,4:124,4:124]);labels.append(z["mask"].T[4:124,4:124].copy());valid.append(v[4:124,4:124].copy())
        images=np.stack(images);labels=np.stack(labels);valid=np.stack(valid);gsd=10
        assert sum(r["split"]=="train" for r in rows)==2223 and sum(r["split"]=="val" for r in rows)==669
        provenance=dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_role="fixed event_ood_v4r1 auxiliary labels; no encoder optimization or selection",
                        orientation="image, target and valid transposed fromx,y together",crop_source_pixels=[4,4,120,120],
                        boundary="cached10m common-grid inputs; 20m band inputs reconstructed by area aggregation for this diagnostic; same members but1200m crop differs from old1280m reports")
    valid=valid.astype(bool)&np.isfinite(images).all(1)
    images=np.where(valid[:,None],images,0).astype(np.float32)
    return rows,images,labels,valid,gsd,provenance


def input_grid(image,valid,source_gsd,target_gsd):
    size=1200//target_gsd
    mode="area" if target_gsd>=source_gsd else "bilinear"
    kwargs={} if mode=="area" else dict(align_corners=False)
    coverage=F.interpolate(valid[:,None].float(),size=(size,size),mode=mode,**kwargs)
    value=F.interpolate(image*valid[:,None],size=(size,size),mode=mode,**kwargs)/coverage.clamp_min(1e-6)
    return value,coverage[:,0]>=.8


def build_encoder(kind,checkpoint,statistics):
    if kind in ["transformer","cnn","random"]:
        state=torch.load(checkpoint,map_location="cpu",weights_only=False) if checkpoint else None
        model=make_model(kind,coordinate_mode=checkpoint_coordinate_mode(state) if state else "legacy_local").cuda()
        if kind!="random" and state is None:raise ValueError("trained encoder requires a preselected unlabeled checkpoint")
        if state:
            if state["statistics"]!=statistics:raise ValueError("wrong pretraining normalization statistics")
            model.load_state_dict(state["model"])
        def forward(image,valid,gsd,dates):
            grids={g:input_grid(image,valid,gsd,g) for g in [10,20]}
            bands=[]
            for name,wavelength,g in S2:
                x,v=grids[g];i=NAMES.index(name);stats=statistics["S2:"+name]
                bands.append(BandRaster("S2:"+name,(x[:,i]-stats["mean"])/stats["std"],v,wavelength,None,g,(g/2,g/2)))
            rep=model.encode(bands);weight=rep.valid[...,None].float()
            return ((rep.tokens*weight).sum(1)/weight.sum(1).clamp_min(1)).permute(0,3,1,2)
        info=dict(kind=kind,checkpoint=checkpoint,step=state.get("step") if state else None,
                  encoder_parameters=sum(p.numel() for p in model.tokenizer.parameters())+sum(p.numel() for p in model.encoder.parameters()),
                  readout="valid-band mean after shared encoder; no pre-attention band averaging in Transformer")
    elif kind=="legacy":
        sys.path.insert(0,str(ROOT.parent/"landslide-seg/src"))
        from scripts.frozen_segmentation_probe import encoder
        if not checkpoint:raise ValueError("historical encoder checkpoint must be explicitly selected")
        model,legacy,info=encoder(checkpoint)
        def forward(image,valid,gsd,dates):
            x,v=input_grid(image,valid,gsd,10)
            return legacy(torch.where(v[:,None],x,0))[-1]
        info["boundary"]="historical CNN-DINO; prior region exposure disclosed separately, not assumed isolated from current evaluation geography"
    else:
        path=ROOT.parent/"landslide-seg/scripts/native_official_probe.py"
        spec=importlib.util.spec_from_file_location("observation_official_models",path)
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        if kind=="anysat":
            # Both repositories use `src`; retain this project's packages while resolving official networks first.
            for package,folder in [("src",official.ANYSAT_CODE/"src"),("src.models",official.ANYSAT_CODE/"src/models")]:
                module=importlib.import_module(package)
                module.__path__=[str(folder),*list(module.__path__)]
        model,info=(official.build_dofa if kind=="dofa" else official.build_anysat)(torch.device("cuda"))
        if kind=="dofa":
            wavelengths={name:wave/1000 for name,wave,_ in S2}
            def forward(image,valid,gsd,dates):
                x,v=input_grid(image,valid,gsd,10);x=torch.where(v[:,None],x,0)
                return model(F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False),[wavelengths[n] for n in NAMES])[-1]
            info["preprocessing"]="physical reflectance; same1200m footprint resized224x224; all10 real band channels"
        else:
            from landslide_seg.data.band_mapping import normalize_anysat,ANY_SAT_S2_BANDS
            assert list(ANY_SAT_S2_BANDS)==NAMES
            def forward(image,valid,gsd,dates):
                x,v=input_grid(image,valid,gsd,10)
                x=normalize_anysat(x*10000.)
                x=torch.where(v[:,None],x,0)
                days=torch.tensor([[datetime.fromisoformat(d.replace("Z","+00:00")).timetuple().tm_yday] for d in dates],device=x.device)
                return model(dict(s2=x[:,None],s2_dates=days),patch_size=40,output="dense",output_modality="s2").permute(0,3,1,2)
            info["preprocessing"]="reflectance multiplied10000 ONCE; official DN mean/std; actual acquisition day-of-year;40m patch; complete context+local dense output"
        info["boundary"]="complete official encoder and local official weights; different architecture, scale and unknown pretraining region exposure"
    model.eval().requires_grad_(False)
    info["all_parameters_frozen"]=not any(p.requires_grad for p in model.parameters())
    return model,forward,info


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--dataset",choices=["worldcover","landslide"],required=True)
    parser.add_argument("--kind",choices=["transformer","cnn","random","legacy","dofa","anysat"],required=True)
    parser.add_argument("--checkpoint");parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    if (args.out/"metadata.json").exists():print("complete frozen features already exist");return
    if (args.out/"features.partial.npy").exists():raise FileExistsError("partial extraction exists; inspect before replacing")
    args.out.mkdir(parents=True,exist_ok=True);rows,images,labels,valid,gsd,provenance=auxiliary_data(args.dataset)
    stats=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    model,forward,info=build_encoder(args.kind,args.checkpoint,stats);before=model_hash(model)
    start=time.perf_counter();features=None;torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(0,len(rows),2):
            image=torch.from_numpy(images[i:i+2]).cuda();v=torch.from_numpy(valid[i:i+2]).cuda()
            value=forward(image,v,gsd,[r["date"] for r in rows[i:i+2]])
            value=F.interpolate(value.float(),size=(40,40),mode="bilinear",align_corners=False).half().cpu().numpy()
            if not np.isfinite(value).all():raise FloatingPointError("nonfinite frozen feature")
            if features is None:features=np.lib.format.open_memmap(args.out/"features.partial.npy",mode="w+",dtype=np.float16,shape=(len(rows),*value.shape[1:]))
            features[i:i+len(value)]=value
            if i%20==0:print(json.dumps(dict(dataset=args.dataset,kind=args.kind,observations=i+len(value),total=len(rows))),flush=True)
    features.flush();shape=list(features.shape);del features
    after=model_hash(model)
    if before!=after:raise ValueError("frozen encoder parameters or buffers changed")
    (args.out/"features.partial.npy").replace(args.out/"features.npy")
    np.savez_compressed(args.out/"targets.npz",label=labels,valid=valid)
    (args.out/"members.json").write_text(json.dumps(rows))
    result=dict(status="frozen_features_complete",dataset=args.dataset,encoder=info,encoder_state_sha256_before=before,encoder_state_sha256_after=after,
                source=provenance,feature_shape=shape,feature_storage="float16, converted toFP32 for probe; all methods on40x40 common30m readout grid",
                input_window_m=1200,target_grid_m=gsd,seconds=time.perf_counter()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
                gpu_name=torch.cuda.get_device_name(),cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES","all; usingcuda:0"),
                checkpoint_sha256=info.get("checkpoint_sha256") or (hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None),
                selection="encoder fixed before auxiliary labels; only task heads may be selected on their validation split")
    (args.out/"metadata.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=="__main__":main()
