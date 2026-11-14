"""Index every completed V3-FR run and draw from its saved raw prediction arrays."""
import argparse
from contextlib import ExitStack
import csv
from html import escape
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("campaign", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
campaign = args.campaign.resolve()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
figures = out / "figures"


def read(path):
    return json.loads(path.read_bytes())


names = read(campaign/"experiment_display_names.json")


def row_label(ax, arm):
    label = names[arm]["name"].replace(" + ", "\n+ ", 1)
    ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=12, fontsize=10, weight="bold")


with (campaign/"experiment_registry.csv").open(encoding="utf-8-sig", newline="") as f:
    registry = list(csv.DictReader(f))
fits = [r["id"] for r in registry if r["stage"]=="S5"]
trials = [r["id"] for r in registry if r["stage"]=="S6" and
          (campaign/"s6_component_trials"/r["id"]/"status.json").is_file()]
assert trials==["B0","BP","N0","N1","N2","C"] and fits==["AE-L","AE-F","OR-D0","OR-D1"]
plt.rcParams.update({"font.family":["Microsoft YaHei","DejaVu Sans"],"font.size":10})
gray = plt.colormaps["gray"].with_extremes(bad="#d7e7ef")
groups = []


def draw(ax, values, limits, cmap=gray, border=False):
    ax.imshow(values,interpolation="nearest",cmap=cmap,norm=Normalize(*limits))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border)
        if border:
            spine.set_color("#8899a5")


def finish(fig, name, pdf):
    path = out / (name+".png")
    fig.savefig(path,dpi=140,facecolor="white")
    pdf.savefig(fig,facecolor="white")
    plt.close(fig)
    return path


def diff_limits(fields, masks):
    valid = [np.abs(field[mask]) for field,mask in zip(fields,masks,strict=True)]
    bound = max(float(np.quantile(np.concatenate(valid),.98)),1e-8)
    return 0,bound


def same(a,b):
    assert a.shape==b.shape and np.array_equal(a,b,equal_nan=True)


def trial_grid(arrays, obs, sensor, task, pdf, step):
    base = arrays["B0"]
    marker = f"{obs}_{task}_"
    prefixes = sorted(k.removesuffix("_target_mask") for k in base.files
                      if k.startswith(marker) and k.endswith("_target_mask"))
    p = max(prefixes,key=lambda k:int(base[k+"_target_mask"].sum()))
    band = p.removeprefix(marker)
    truth,qa,keep,target,limits = (base[p+"_"+k] for k in
        ("truth","qa","visible_mask","target_mask","display_scale"))
    if task=="A":
        reference = np.where(keep,truth,np.nan)
        ref_name,ref_limits = band,limits
    else:
        references = sorted(k for k in base.files if k.startswith(marker+"visible_reference_"))
        key = references[0]
        reference = base[key]
        ref_name,ref_limits = key.split("_visible_reference_")[1],np.nanquantile(reference,[.02,.98])
    predictions, contributions, errors = {},{},{}
    for arm,raw in arrays.items():
        for k,v in (("truth",truth),("target_mask",target),("visible_mask",keep),("display_scale",limits)):
            same(raw[p+"_"+k],v)
        main,detail,total = (raw[p+"_"+k] for k in ("main_prediction","detail_contribution","total_prediction"))
        np.testing.assert_allclose((main+detail)[target],total[target],atol=1e-6,rtol=1e-6)
        if task=="A":
            assert not np.any(detail[target])
        predictions[arm] = main,total
        contributions[arm] = detail
        errors[arm] = abs(total-truth)
    span = diff_limits(list(contributions.values()),[target]*len(trials))[1]
    error_range = diff_limits(list(errors.values()),[target]*len(trials))
    fig = plt.figure(figsize=(10.6,12.4))
    fig.subplots_adjust(left=.29,right=.975,top=.86,bottom=.08,hspace=.30,wspace=.15)
    grid = fig.add_gridspec(7,4,height_ratios=[.95,1,1,1,1,1,1])
    for j,(image,title,scale) in enumerate(((np.where(qa,truth,np.nan),"原生真值",limits),
          (reference,f"实际可见输入 · {ref_name}",ref_limits),(target,"目标掩码（白色为目标）",(0,1)))):
        ax = fig.add_subplot(grid[0,j])
        draw(ax,image,scale,border=j==2)
        ax.set_title(title,pad=3)
    for i,arm in enumerate(trials,1):
        main,total = predictions[arm]
        for j,(image,title,scale,cmap) in enumerate([
            (main,"共享主路径输出",limits,gray),(total,"总输出",limits,gray),
            (contributions[arm],"历史 B1 细节支路贡献",(-span,span),plt.colormaps["RdBu_r"].with_extremes(bad="#d7e7ef")),
            (errors[arm],"总输出绝对误差",error_range,plt.colormaps["magma"].with_extremes(bad="#d7e7ef"))]):
            ax = fig.add_subplot(grid[i,j])
            draw(ax,image,scale,cmap)
            if i==1:
                ax.set_title(title,pad=3)
            if j==0:
                row_label(ax,arm)
    fig.suptitle(f"S6 · {task} · {sensor} · {band} · 第 {step} 步",y=.977,fontsize=19,weight="bold")
    fig.text(.105,.933,obs,fontsize=10)
    fig.text(.105,.905,"六种模型架构 · 同观测、同目标掩码、同真值色标",fontsize=11)
    fig.text(.105,.032,"该图选取该观测目标像元数最多的波段。浅蓝色为无预测或无效像元。误差与支路贡献各用共享色标。",
             fontsize=9)
    return finish(fig,f"S6_{step:03d}_{sensor}_{task}",pdf)


def fit_grid(obs,sensor,pdf,role="fit"):
    arms = fits if role=="fit" else ["AE-L","AE-F"]
    with ExitStack() as stack:
        def values(arm,step):
            p = campaign/"s5_capacity_fit"/arm/f"predictions_{step:03d}.npz"
            return stack.enter_context(np.load(p,allow_pickle=False))
        initial = {arm:values(arm,0) for arm in arms}
        final = {arm:values(arm,400) for arm in arms}
        p = sorted(k.removesuffix("_mask") for k in final["AE-L"].files
                   if k.startswith(obs+"_") and k.endswith("_mask"))
        key = max(p,key=lambda k:int(final["AE-L"][k+"_mask"].sum()))
        band = key.removeprefix(obs+"_")
        truth,mask = (final["AE-L"][key+"_"+k] for k in ("truth","mask"))
        limits = np.quantile(truth[mask],[.02,.98])
        outputs = {}
        for arm in arms:
            for raw in (initial[arm],final[arm]):
                same(raw[key+"_truth"],truth)
                same(raw[key+"_mask"],mask)
            outputs[arm] = tuple(raw[key+"_prediction"] for raw in (initial[arm],final[arm]))
        errors = [abs(outputs[arm][1]-truth) for arm in arms]
        error_range = diff_limits(errors,[mask]*len(arms))
        fig = plt.figure(figsize=(10.6,9 if role=="fit" else 5.6))
        fig.subplots_adjust(left=.29,right=.97,top=.82 if role=="fit" else .76,bottom=.075,hspace=.28,wspace=.18)
        grid = fig.add_gridspec(len(arms),4)
        for i,arm in enumerate(arms):
            for j,(values,title,scale,cmap) in enumerate([
                (np.where(mask,truth,np.nan),"原生真值",limits,gray),
                (outputs[arm][0],"第 0 步输出",limits,gray),
                (outputs[arm][1],"第 400 步输出",limits,gray),
                (errors[i],"第 400 步绝对误差",error_range,plt.colormaps["magma"].with_extremes(bad="#d7e7ef"))]):
                ax = fig.add_subplot(grid[i,j])
                draw(ax,values,scale,cmap)
                if i==0:
                    ax.set_title(title,pad=3)
                if j==0:
                    row_label(ax,arm)
        kind = "拟合样本" if role=="fit" else "只读样本"
        fig.suptitle(f"S5 · {sensor} · {band} · TRAIN {kind}",y=.968,fontsize=19,weight="bold")
        fig.text(.126,.916,obs,fontsize=10)
        fig.text(.126,.865,"统一展示第 400 步 · 历史 B1 细节支路关闭",fontsize=10)
        return finish(fig,f"S5_{sensor}" if role=="fit" else f"S5_read_only_{sensor}",pdf)


def paired_grid(task, pdf):
    folder = campaign/"s6_component_trials/paired_examples"
    cases = [case for case in read(folder/"selection.json")["examples"] if case["task"]==task]
    assert len(cases)==3
    fig, axes = plt.subplots(3,7,figsize=(15,6.7))
    fig.subplots_adjust(left=.095,right=.99,top=.81,bottom=.05,wspace=.14,hspace=.25)
    titles = ["原生真值","可见输入","目标掩码","D0 连续插值\n共享主路径","D1 子像素\n共享主路径",
              "D1 子像素\n总输出","D1 子像素\n总输出绝对误差"]
    labels = ("改善最大","中间案例","退化最大")
    for i,(case,label) in enumerate(zip(cases,labels,strict=True)):
        with np.load(folder/case["float_file"],allow_pickle=False) as raw:
            truth,qa,mask,limits = (raw[k] for k in ("truth","qa","target_mask","display_scale"))
            reference = raw["visible_reference"] if task=="B1" else np.where(raw["visible_mask"],truth,np.nan)
            ref_limits = np.nanquantile(reference,[.02,.98]) if task=="B1" else limits
            a,b = (raw[k] for k in ("N2_main_prediction","C_main_prediction"))
            left,right = (raw[k] for k in ("N2_predictions","C_predictions"))
            error_range = diff_limits([left-truth,right-truth],[mask,mask])
            images = [(np.where(qa,truth,np.nan),limits),(reference,ref_limits),(mask,(0,1)),
                      (a,limits),(b,limits),(right,limits),(abs(right-truth),error_range)]
            for j,(ax,(image,scale)) in enumerate(zip(axes[i],images,strict=True)):
                draw(ax,image,scale,plt.colormaps["magma"].with_extremes(bad="#d7e7ef") if j==6 else gray,border=j==2)
                if i==0:
                    ax.set_title(titles[j],fontsize=10)
            axes[i,0].set_ylabel(f"{label}\n{case['sensor']} {case['band']}\nMSE Δ {case['mse_delta']:+.4f}",
                                 fontsize=9,rotation=0,ha="right",va="center",labelpad=8)
    fig.suptitle(f"FiLM 浅层 ResNet · D0 连续插值与 D1 子像素解码 · {task} · 第 233 步",fontsize=17)
    return finish(fig,f"s6_D1_paired_{task}",pdf)


panel = read(campaign/"data_panel.json")
details = []
for path in sorted(figures.glob("*.png")):
    match = re.fullmatch(r"s6_(BP|N0|N1|N2|C)_(obs_\w+)_(A|B1)",path.stem)
    if match:
        arm,obs,task = match.groups()
        details.append((f"{names[arm]['name']} · {task} · {obs}",path))
    elif path.stem.endswith("_historical"):
        details.append((f"{names['S1-REPLAY']['name']} · {path.stem.removesuffix('_historical')}",path))
assert len(details)==30
paired = []


def add_page(pdf,path):
    im = plt.imread(path)
    fig = plt.figure(figsize=(11,11*im.shape[0]/im.shape[1]))
    ax = fig.add_axes((0,0,1,1))
    ax.imshow(im)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


with PdfPages(out/"V3-FR.pdf") as pdf:
    for name in ("s6_overview.png","s5_fit.png"):
        add_page(pdf,figures/name)
    for task in ("A","B1"):
        paired.append((f"FiLM 浅层 ResNet · D1 与 D0 配对 · {task}",paired_grid(task,pdf)))
    for step in (233,0):
        with ExitStack() as stack:
            arrays = {arm:stack.enter_context(np.load(campaign/"s6_component_trials"/arm/
                      f"val_{step:03d}/raw_predictions.npz",allow_pickle=False)) for arm in trials}
            for obs,sensor in zip(panel["fixed_val"],("S2","L8"),strict=True):
                for task in ("A","B1"):
                    path = trial_grid(arrays,obs,sensor,task,pdf,step)
                    groups.append((f"S6 · {sensor} · {task} · 第 {step} 步",path))
    for row in panel["fit"]:
        path = fit_grid(row["id"],row["sensor"],pdf)
        groups.append((f"S5 · {row['sensor']} · 第 400 步",path))
    for row in panel["read_only"]:
        path = fit_grid(row["id"],row["sensor"],pdf,"read_only")
        groups.append((f"S5 · {row['sensor']} · 只读样本 · 第 400 步",path))
    for _,path in details:
        add_page(pdf,path)


def href(path):
    return escape(Path(os.path.relpath(path,out)).as_posix(),quote=True)


def tile(label,path):
    url = href(path)
    return f'<figure><a href="{url}"><img loading="lazy" src="{url}" alt="{escape(label)}"></a><figcaption>{escape(label)}</figcaption></figure>'


run_table = "".join(f"<tr><th scope='row'>{escape(row['name'])}</th><td>{escape(row['architecture'])}</td><td>{key}</td></tr>"
                    for key,row in names.items())
doc = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V3-FR 实验图集</title>
<style>
body{font:16px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:#172b3a;background:#f3f6f8;margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 18px 64px}
a{color:#18567b} h2{margin-top:38px} figure{background:white;margin:0;padding:10px;border:1px solid #d5e0e6}
img{display:block;width:100%;height:auto} figcaption{font-size:14px;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,410px),1fr));gap:16px}
table{border-collapse:collapse;background:white} th,td{text-align:left;padding:5px 12px;border-bottom:1px solid #dde6eb}
nav{display:flex;gap:1em;flex-wrap:wrap}
</style><main>
<h1>V3-FR 实验图集</h1>
<p>FiLM 浅层 ResNet + D1 子像素解码的 B1 总输出是共享主路径输出加上历史 B1 细节支路贡献。A 的细节支路贡献为零。</p>
<p>S6 各组使用 seed 17，每组 233 步，验证节点为 0、42、116、233。重建图使用保存的第 0 步和第 233 步浮点数组。S5 的自由 Latent 组只在两条拟合样本上有输出。</p>
<nav><a href="#curves">验证与拟合曲线</a><a href="#s6">S6 固定样本</a><a href="#s5">S5 拟合样本</a><a href="#detail">逐对照与历史复算</a><a href="V3-FR.pdf">PDF</a></nav>
<h2>实验模型与架构</h2><table><thead><tr><th>实验模型</th><th>完整结构</th><th>追溯 ID</th></tr></thead><tbody>""" + run_table + """</tbody></table>
<p>S7 多种子确认未触发。</p>
<h2 id="curves">验证与拟合曲线</h2><div class="grid">""" + "".join(
    tile(label,figures/name) for label,name in (("S6 · 六组验证曲线与配对结果","s6_overview.png"),
                                                ("S5 · 四组拟合曲线","s5_fit.png"))) + "</div>"
for step in (233,0):
    doc += f'<h2 id="s6{"" if step==233 else "-initial"}">S6 · 第 {step} 步</h2><div class="grid">'
    doc += "".join(tile(label,path) for label,path in groups if path.name.startswith(f"S6_{step:03d}_"))
    doc += "</div>"
doc += '<h2 id="s5">S5 · 两条拟合样本与四条只读样本</h2><div class="grid">'
doc += "".join(tile(label,path) for label,path in groups if path.name.startswith("S5_"))
doc += '</div><h2 id="detail">逐对照与历史复算</h2><div class="grid">'
doc += "".join(tile(label,path) for label,path in paired+details) + "</div></main></html>"
(out/"index.html").write_text(doc,encoding="utf-8")
markdown = [
    "# V3-FR 实验图集",
    "",
    "[报告](../summary/REPORT.md) · [完整 PDF](V3-FR.pdf)",
    "",
    "B1 总输出等于共享主路径输出加上历史 B1 细节支路贡献。A 的支路贡献为零。",
    "",
    "| 实验模型 | 完整结构 | 追溯 ID |",
    "| --- | --- | --- |",
]
for key,row in names.items():
    markdown.append(f"| {row['name']} | {row['architecture']} | `{key}` |")
markdown += ["", "## 验证与拟合曲线", "",
             "![S6 六种模型架构的验证曲线与配对结果](figures/s6_overview.png)", "",
             "![S5 四种模型架构的拟合曲线](figures/s5_fit.png)"]
for heading, section in (("S6 固定样本", [item for item in groups if item[1].name.startswith("S6_")]),
                         ("S5 拟合与只读样本", [item for item in groups if item[1].name.startswith("S5_")]),
                         ("逐对照与历史复算", paired+details)):
    markdown += ["",f"## {heading}",""]
    markdown += [f"- [{label}]({href(path)})" for label,path in section]
(out/"README.md").write_text("\n".join(markdown)+"\n",encoding="utf-8")
print(f"Indexed {len(groups)+len(details)+len(paired)+2} figures with architecture labels")
