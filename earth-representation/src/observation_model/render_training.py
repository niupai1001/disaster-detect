"""Real session curves and logged compute exposure, including replay after rollback."""
import json
import os
from pathlib import Path
from datetime import datetime,timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

from .data import ROOT

OUT=Path(os.environ.get("OBSERVATION_OUTPUT_ROOT",ROOT/"observation_model/outputs"))
NAMES={"masked_transformer":"共享光谱—空间Transformer", "masked_cnn":"同容量CNN编码器", "masked_random":"冻结随机编码器",
       "cross_masked_control":"同源继续训练对照", "cross_cross30":"加入双向跨源监督", "observation_none":"跨源继续训练｜无空间约束", "observation_fixed":"跨源训练｜有空间观测约束"}


def read_sessions(run):
    path=OUT/"tvlp"/(os.environ.get("OBSERVATION_RUN_PREFIX","obs_")+run)/"metrics.jsonl"
    if not path.exists():path=OUT/"tvlp"/("obs_"+run)/"metrics.jsonl"
    if not path.exists():return {},[]
    sessions={};invalid=[];lines=path.read_text().splitlines()
    for i,line in enumerate(lines):
        try:row=json.loads(line)
        except json.JSONDecodeError:
            invalid.append(dict(line=i+1,characters=len(line)));continue
        item=sessions.setdefault(row["session_id"],dict(train={},validation=[],timestamps=[]))
        item["timestamps"].append(datetime.fromisoformat(row["timestamp"]))
        if row["phase"]=="train":item["train"].setdefault(row["global_step"],{}).update(row["metrics"])
        elif row["phase"]=="val" and "score" in row["metrics"]:item["validation"].append((row["global_step"],row["metrics"]["score"]))
    return sessions,invalid


def summarize(sessions):
    if not sessions:return None
    latest=next(reversed(sessions.values()));last=max(latest["train"],default=0)
    logged=sum(len(s["train"]) for s in sessions.values())
    unique_steps={step for s in sessions.values() for step in s["train"]}
    peaks=[r["gpu_peak_mib"] for s in sessions.values() for r in s["train"].values() if "gpu_peak_mib" in r]
    durations=[max(0.,(s["timestamps"][-1]-s["timestamps"][0]).total_seconds()) for s in sessions.values()]
    return dict(latest_session_step=last,logged_optimizer_updates=logged,logged_repeated_updates=logged-len(unique_steps),
                observation_exposure_logged=logged*32,session_count=len(sessions),logged_active_wall_seconds=sum(durations),
                peak_gpu_mib=max(peaks) if peaks else None,
                cost_definition="unique(session_id, global_step) train records; active wall time includes validation and I/O, excludes offline gaps and unlogged startup; not exclusive GPU time")


def main():
    font=ROOT.drive+"/Windows/Fonts/msyh.ttc" if ROOT.drive.lower()=="c:" else "C:/Windows/Fonts/msyh.ttc"
    from pathlib import Path
    if Path(font).exists():plt.rcParams["font.family"]=FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"]=False
    row_count=3 if any((OUT/"tvlp"/("obs_"+name)/"metrics.jsonl").exists() for name in ["observation_none","observation_fixed"]) else 2
    fig,axes=plt.subplots(row_count,2,figsize=(14,4.5*row_count),constrained_layout=True)
    colors=plt.get_cmap("tab10");costs={}
    for index,(name,title) in enumerate(NAMES.items()):
        sessions,invalid=read_sessions(name);summary=summarize(sessions)
        if summary is None:continue
        summary["unreadable_metric_records"]=invalid
        costs[name]=summary;row=0 if name.startswith("masked_") else 2 if name.startswith("observation_") else 1;color=colors(index)
        loss_title=title+({"cross_masked_control":"（仅整波段损失）","cross_cross30":"（整波段＋跨源损失）","observation_none":"（整波段＋跨源损失）","observation_fixed":"（再加观测损失）"}.get(name,""))
        for j,item in enumerate(sessions.values()):
            alpha=1. if j==len(sessions)-1 else .4
            points=[(step,r["loss"]) for step,r in item["train"].items() if "loss" in r]
            # Nonoverlapping100-update means within each session; no smoothing across a rollback.
            points=[points[k:k+100] for k in range(0,len(points),100)]
            if points:
                x=[np.mean([p[0] for p in block]) for block in points];y=[np.mean([p[1] for p in block]) for block in points]
                axes[row,0].plot(x,y,color=color,alpha=alpha,label=loss_title if j==len(sessions)-1 else None)
            if item["validation"]:
                x,y=zip(*item["validation"]);axes[row,1].plot(x,y,color=color,alpha=alpha,label=title if j==len(sessions)-1 else None)
    for row,label in enumerate(["同源整波段恢复","真实双向跨源预测","空间观测约束比较"][:row_count]):
        for col in range(2):
            ax=axes[row,col];ax.set_title(label+("｜训练损失" if col==0 else "｜无标签验证评分"));ax.set_xlabel("本阶段逻辑更新步数")
            ax.set_ylabel("Huber损失（每会话100步均值）" if col==0 else "传感器/方向等权归一化RMSE")
            ax.set_yscale("log");ax.grid(alpha=.2);ax.legend(fontsize=9)
    for row,path,key in [(0,OUT/"evaluation/val/masked_ridge.json","singleton"),(1,OUT/"phase1/evaluation/ridge_cross_fixedqa.json","cross30")]:
        if path.exists():
            value=json.loads(path.read_text())["scores"][key]["score"]
            axes[row,1].axhline(value,color="black",linestyle="--",label="Ridge验证参照");axes[row,1].legend(fontsize=9)
        else:
            baseline=OUT/"baselines"/("masked_validation.json" if row==0 else "cross_validation.json")
            if baseline.exists():
                value=json.loads(baseline.read_text())["results"]["ridge"]["score"]
                axes[row,1].axhline(value,color="black",linestyle="--",label="同版本Ridge验证参照");axes[row,1].legend(fontsize=9)
    fig.suptitle("运行中快照｜浅色为较早会话，深色为最新会话；恢复后的重算不抹去")
    folder=OUT/"final/figures";folder.mkdir(exist_ok=True,parents=True);fig.savefig(folder/"training_curves.png",dpi=160);plt.close(fig)
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),status="live_snapshot_not_final_acceptance",runs=costs,
                logged_total_updates=sum(r["logged_optimizer_updates"] for r in costs.values()),
                logged_repeated_updates=sum(r["logged_repeated_updates"] for r in costs.values()),
                boundary="logged updates are a lower bound on actual work if a crash occurred before logging; inference, probes and engineering costs are separate")
    (OUT/"final/training_costs_live.json").write_text(json.dumps(result,indent=2));print(json.dumps({k:v["latest_session_step"] for k,v in costs.items()}))


if __name__=="__main__":main()
