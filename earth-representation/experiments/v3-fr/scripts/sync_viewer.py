"""Mirror the new V3-FR metric journals into the existing Training Viewer."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def entries(path, offset):
    if not path.exists():
        return
    with path.open("rb") as stream:
        stream.seek(offset)
        for line in stream:
            if not line.endswith(b"\n"):
                break
            yield stream.tell(),json.loads(line)


def loss(label):
    return dict(label=label,unit="z²",direction="min",role="loss")


def writer_for(root,path,group,binding,writer):
    is_fit = group=="fit"
    name = path.name
    charts = (["train","fit","held"] if is_fit else ["train","val_A","val_B1","main_B1"])
    metrics = dict(train=loss("工程训练目标"),fit=loss("拟合样本 MSE"),held=loss("只读 TRAIN MSE"),
                   val_A=loss("A · 完整 VAL MSE"),val_B1=loss("B1 · 完整 VAL MSE"),
                   main_B1=loss("B1 · 共享主路径 MSE"),texture_A=loss("A · 高纹理梯度 MSE"),
                   main_texture_B1=loss("B1 · 主路径高纹理梯度 MSE"))
    names = dict(B0="历史前端 D0",BP="物理 Q/K 位置",N0="有序前端与空间融合",
                 N1="浅层 ResNet",N2="FiLM ResNet",C="完整主路径与 D1")
    display = names.get(binding.get("experiment_id"),name)
    target = root/("v3_fr_20261113_"+name.lower())
    if (target/"run.json").exists():
        return None
    return writer(target,
        experiment=dict(id="encoder-v3-fr-"+group,name="Encoder V3-FR · "+("全可见拟合" if is_fit else "模块对照"),
                        description="固定历史 Transformer 的有序打包、FiLM ResNet、空间融合和 D1"),
        variant=dict(id=name.lower(),name=display,kind="baseline" if name in ("AE-L","B0") else "modification",
                     description=binding.get("changed_factor",display)),
        metrics=metrics,charts=charts,numbers=[] if is_fit else ["texture_A","main_texture_B1"],
        primary_metric="fit" if is_fit else "val_A",config=dict(source=str(path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign",type=Path)
    parser.add_argument("viewer_root",type=Path)
    parser.add_argument("writer",type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("tvlp_writer",args.writer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    publishers,skipped = {},set()
    groups = {"s5_capacity_fit":"fit","s6_component_trials":"ab","s7_seed_confirmation":"confirm"}
    stop = args.campaign/"VIEWER_SYNC_STOP"
    try:
        while True:
            for folder,group in groups.items():
                for path in sorted((args.campaign/folder).glob("*/identity.json")):
                    run = path.parent
                    if run in skipped:
                        continue
                    if run not in publishers:
                        binding = read(path)
                        if not binding:
                            continue
                        writer = writer_for(args.viewer_root,run,group,binding,module.RunWriter)
                        if writer is None:
                            skipped.add(run)
                            continue
                        publishers[run] = dict(writer=writer,offset=0,nodes=set(),terminal=False)
                    state = publishers[run]
                    if state["terminal"]:
                        continue
                    queue = []
                    for offset,row in entries(run/"updates.jsonl",state["offset"]):
                        step = row.get("cursor",row.get("step"))
                        score = row["loss"] if "loss" in row else sum(row["losses"].values())/len(row["losses"])
                        queue.append((step,{"train":score}))
                        state["offset"] = offset
                    if group=="fit":
                        by_step = defaultdict(lambda:defaultdict(list))
                        for row in read(run/"evaluation.json") or []:
                            if row["step"] not in state["nodes"]:
                                by_step[row["step"]][row["role"]].append(row["mse"])
                        for step,roles in by_step.items():
                            queue.append((step,{("fit" if k=="fit" else "held"):sum(v)/len(v) for k,v in roles.items()}))
                            state["nodes"].add(step)
                    else:
                        for node in run.glob("val_*/summary.json"):
                            summary = read(node)
                            if not summary or summary["step"] in state["nodes"]:
                                continue
                            step,values = summary["step"],summary["groups"]
                            a,b = values["A/all"],values["B1/all"]
                            queue.append((step,dict(val_A=a["mse"],val_B1=b["mse"],main_B1=b["main_path_mse"],
                                texture_A=a["texture_gradient_mse"],main_texture_B1=b["main_texture_gradient_mse"])))
                            state["nodes"].add(step)
                    writer = state["writer"]
                    for step,values in sorted(queue,key=lambda row:row[0]):
                        if step<writer.step:
                            # A validation node can be made durable after its matching update.
                            writer.step = step
                        writer.log(step,values)
                    status = read(run/"status.json")
                    if status and status["status"] in ("COMPLETE","BLOCKED","STOPPED"):
                        writer.finish(failed=status["status"]=="BLOCKED")
                        state["terminal"] = True
            index = {run.name:dict(run_id=s["writer"].meta["run_id"],directory=str(s["writer"].path))
                     for run,s in publishers.items()}
            tmp = args.campaign/"viewer_index.tmp"
            tmp.write_text(json.dumps(index,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            tmp.replace(args.campaign/"viewer_index.json")
            if stop.exists():
                break
            time.sleep(6)
    finally:
        for state in publishers.values():
            if not state["terminal"]:
                state["writer"].stop.set()
                state["writer"].lock_file.close()


if __name__=="__main__":
    main()
