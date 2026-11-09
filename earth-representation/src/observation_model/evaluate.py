"""Fixed target-QA evaluation, shared by neural and simple predictors; TEST is explicit."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


from .baselines import OUT, SpectralControl
from .data import RELEASE, NativeSet
from .model import make_model, checkpoint_coordinate_mode
from .paired import NativePairs
from .train_cross import cross_query


def correlation(x, y):
    x=x.double()-x.double().mean();y=y.double()-y.double().mean()
    scale=(x.square().sum()*y.square().sum()).sqrt()
    return float((x*y).sum()/scale) if scale>1e-12 else None


def midranks(x):
    _, inverse, counts=torch.unique(x, sorted=True, return_inverse=True, return_counts=True)
    ranks=counts.cumsum(0).double()-(counts.double()+1)/2
    return ranks[inverse]


def measure(prediction, truth, valid, coverage):
    # Every method uses the same measured target QA; predicted coverage never removes errors.
    x=prediction[valid].double();y=truth[valid].double()
    if not len(x) or not torch.isfinite(x).all():
        raise ValueError("empty measured target or nonfinite prediction on a measured target pixel")
    error=x-y
    nonflat=len(y)>1 and float(y.std(unbiased=False))>=.005
    return dict(count=len(x),sum_absolute=float(error.abs().sum()),sum_squared=float(error.square().sum()),
                sum_error=float(error.sum()),uncovered_target_pixels=int((valid&~coverage).sum()),
                truth_std=float(y.std(unbiased=False)),nonflat=nonflat,
                pearson=correlation(x,y) if nonflat else None,
                spearman=correlation(midranks(x),midranks(y)) if nonflat else None)


def summarize(records, statistics):
    groups={}
    for row in records:
        key=(row["mask_set"],row["direction"],row["band"])
        groups.setdefault(key,[]).append(row)
    metrics=[]
    for (mask_set,direction,band), members in groups.items():
        n=sum(r["count"] for r in members)
        rmse=(sum(r["sum_squared"] for r in members)/n)**.5
        correlations={key:[r[key] for r in members if r[key] is not None] for key in ["pearson","spearman"]}
        metrics.append(dict(mask_set=mask_set,direction=direction,band=band,count=n,scenes=len(members),
                       rmse=rmse,mae=sum(r["sum_absolute"] for r in members)/n,
                       bias=sum(r["sum_error"] for r in members)/n,normalized_rmse=rmse/statistics[band]["std"],
                       uncovered_target_pixels=sum(r["uncovered_target_pixels"] for r in members),
                       nonflat_scene_mean_pearson=float(np.mean(correlations["pearson"])) if correlations["pearson"] else None,
                       nonflat_scene_mean_spearman=float(np.mean(correlations["spearman"])) if correlations["spearman"] else None))
    scores={}
    for mask_set in sorted({r["mask_set"] for r in metrics}):
        direction_scores={direction:float(np.mean([r["normalized_rmse"] for r in metrics if r["mask_set"]==mask_set and r["direction"]==direction]))
                          for direction in sorted({r["direction"] for r in metrics if r["mask_set"]==mask_set})}
        scores[mask_set]=dict(score=float(np.mean(list(direction_scores.values()))),directions=direction_scores)
    return dict(scores=scores,bands=metrics)


def load_predictor(kind, mode, checkpoint, statistics, device):
    if kind in ["mean","interpolation","ridge"]:
        fits=torch.load(OUT/(mode+"_ridge_fits.pt"),map_location=device,weights_only=False) if kind=="ridge" else None
        lambdas=json.loads((OUT/(mode+"_ridge_lambdas.json")).read_text()) if kind=="ridge" else None
        return SpectralControl(statistics,kind,fits,lambdas,mode),dict(method=kind,selected_on="VAL only")
    state=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if kind=="mlp":
        from .pixel_mlp import PixelMLP
        model=PixelMLP(statistics)
    else:
        model=make_model(kind,coordinate_mode=checkpoint_coordinate_mode(state))
    if state.get("statistics") is not None and state["statistics"]!=statistics:
        raise ValueError("checkpoint statistics differ from the frozen data release")
    model.load_state_dict(state.get("model",state));model.to(device).eval()
    return model,dict(method=kind,checkpoint=str(Path(checkpoint).resolve()),step=state.get("step"))


@torch.inference_mode()
def evaluate(model, data, statistics, mode, mask_set="all", limit_scenes=0, case_dir=None, pair_set="strict"):
    records=[];case_keys=set();contract=json.loads((RELEASE/"masking.json").read_text())
    def record(prediction,truth,valid,coverage,info,source_bands):
        records.append(dict(**info,**measure(prediction,truth,valid,coverage)))
        key=(info["direction"],info["mask_set"])
        if case_dir is not None and key not in case_keys:
            case_keys.add(key);name="_".join(key)
            visible=next(b for b in source_bands if b.valid.any())
            physical=visible.value[0]*statistics[visible.name]["std"]+statistics[visible.name]["mean"]
            np.savez_compressed(case_dir/(name+".npz"), visible_input=physical.cpu().numpy(),visible_valid=visible.valid[0].cpu().numpy(),
                                target=truth.cpu().numpy(),prediction=prediction.float().cpu().numpy(),valid=valid.cpu().numpy(),
                                prediction_coverage=coverage.cpu().numpy(),metadata=json.dumps(dict(**info,visible_input_band=visible.name)))
    if mode=="masked":
        for sensor in ["S2","L8"]:
            c=len(data.metadata[sensor]);masks=[]
            if mask_set in ["all","singleton"]:masks.extend(("singleton",[j]) for j in range(c))
            if mask_set in ["all","unseen"]:masks.extend(("unseen",hidden) for hidden in contract[sensor]["unseen_combinations"])
            count=len(data.records[sensor]);count=min(count,limit_scenes) if limit_scenes else count
            for index in range(count):
                ids=torch.tensor([index],device=data.device);row=data.records[sensor][index]
                for label,hidden in masks:
                    masked=torch.zeros(1,c,device=data.device,dtype=torch.bool);masked[:,hidden]=True
                    inputs=data.bands(sensor,ids,statistics,masked)
                    representation=model.encode(inputs)
                    for j in hidden:
                        query=data.query(sensor,j,ids);prediction,coverage=model.decoder(representation,query)
                        info=dict(id=row["observation_id"],group=row["group"],acquisition=row["acquisition_id"],
                                  direction=sensor+"_same_sensor",band=query.band,mask_set=label,hidden_bands=[data.metadata[sensor][k]["name"] for k in hidden])
                        record(prediction[0],data.value[sensor][query.band][index],data.valid[sensor][query.band][index],coverage[0],info,inputs)
    else:
        if pair_set=="strict":pair_records=None
        else:
            pair_records=[json.loads(line) for line in (RELEASE/"cross30.jsonl").read_text().splitlines()]
            if pair_set=="broad":pair_records=[r for r in pair_records if r["broad_eligible"]]
        pairs=NativePairs(data,pair_records);count=len(pairs.records);count=min(count,limit_scenes) if limit_scenes else count
        for source,target in getattr(model,"source_directions",[("S2","L8"),("L8","S2")]):
            for i,row in enumerate(pairs.records[:count]):
                ids=pairs.indices[source][i:i+1];inputs=data.bands(source,ids,statistics);representation=model.encode(inputs)
                for j,band in enumerate(data.metadata[target]):
                    query=cross_query(data,target,j,pairs.phase[i:i+1]);prediction,coverage=model.decoder(representation,query)
                    info=dict(id=row["site_id"],group=row["group"],acquisition=row["s2_scene"]+"__"+row["landsat_scene"],
                              direction=source+"_to_"+target,band=band["name"],mask_set="cross30",hidden_bands=[],
                              pairing_quality={k:v for k,v in row.items() if k in ["gap_days","ndvi_correlation","ndvi_absolute_difference","registration_residual_m","registration_correlation"]},
                              all17_joint_valid_fraction=float(pairs.joint[i].float().mean()))
                    record(prediction[0],pairs.target[target][i,j],pairs.valid[target][i,j],coverage[0],info,inputs)
    conditions={}
    if mode=="cross":
        for row in records:
            q=row["pairing_quality"];gap=q["gap_days"];residual=q["registration_residual_m"];corr=q["registration_correlation"]
            temporal="same_day" if gap==0 else "1_to2_days" if gap<=2 else "3_to7_days" if gap<=7 else "8_to30_days" if gap<=30 else "over30_days"
            registration="registration_uncertain" if residual is None or residual>15 or corr is None or corr<.6 else "registration_within_threshold"
            ndvi=q["ndvi_correlation"];difference=q["ndvi_absolute_difference"]
            change="NDVI_correlation_undefined" if ndvi is None else "NDVI_change_or_disagreement" if ndvi<.6 or difference>.2 else "NDVI_stable"
            for category in [temporal,registration,change,"nonflat_target" if row["nonflat"] else "flat_target"]:
                conditions.setdefault(category,[]).append(row)
        conditions={key:summarize(value,statistics) for key,value in conditions.items()}
    return dict(**summarize(records,statistics),conditions=conditions,scenes=records,scope=dict(mode=mode,split=data.split,mask_set=mask_set,limit_scenes=limit_scenes,pair_set=pair_set if mode=="cross" else None),
                metric_contract="fixed measured target QA for every predictor; context-uncovered targets retained and counted",
                correlation_contract="scene Pearson and tie-aware Spearman when target reflectance std>=0.005; otherwise marked flat",
                uncertainty_unit="independent acquisition groups; repeated windows are not independent scenes")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["masked","cross"],required=True)
    parser.add_argument("--split",choices=["val","test"],required=True)
    parser.add_argument("--kind",choices=["transformer","cnn","random","mean","interpolation","ridge","mlp"],required=True)
    parser.add_argument("--checkpoint");parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--mask-set",choices=["all","singleton","unseen"],default="all")
    parser.add_argument("--limit-scenes",type=int,default=0)
    parser.add_argument("--pair-set",choices=["strict","broad","all_local"],default="strict",help="broad ignores NDVI filtering; all_local additionally includes long gaps and uncertain registration, diagnostic only")
    parser.add_argument("--route",choices=["direct","observed-fine30"],default="direct")
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    args=parser.parse_args()
    if args.device=="cuda":torch.cuda.init()
    if args.out.exists():raise FileExistsError("preserve prior evaluation; provide a new output path")
    statistics=json.loads((RELEASE/"statistics.json").read_text())["bands"]
    model,provenance=load_predictor(args.kind,args.mode,args.checkpoint,statistics,args.device)
    if args.route=="observed-fine30":
        if args.mode!="cross" or args.kind not in ["transformer","cnn","random"]:raise ValueError("fine observation route requires a neural cross predictor")
        from .observation_constraint import observe_fine_prediction
        network=model
        class ObservedFine:
            source_directions=[("S2","L8")]
            def encode(self,bands):return network.encode(bands)
            def decoder(self,representation,query):
                _,_,value,coverage=observe_fine_prediction(network,representation,query)
                return value,coverage>=.8
        model=ObservedFine()
    provenance["route"]=args.route
    provenance["evaluation_precision"]="FP32 neural inference; FP64 metric accumulation"
    provenance["evaluation_device"]=args.device
    case_dir=args.out.parent/(args.out.stem+"_cases");case_dir.mkdir(exist_ok=True,parents=True)
    result=evaluate(model,NativeSet(args.split,args.device),statistics,args.mode,args.mask_set,args.limit_scenes,case_dir,args.pair_set)
    result["predictor"]=provenance
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result["scores"]))


if __name__=="__main__":main()
