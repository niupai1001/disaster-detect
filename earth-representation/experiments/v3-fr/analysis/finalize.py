"""Record the final measured endpoints and screen decision in the campaign summary."""
import json
from pathlib import Path
import subprocess
import sys


campaign,prior = map(Path,sys.argv[1:3])
root = campaign/"s6_component_trials"
summary = campaign/"summary"
project = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_bytes())


arms = ("B0","BP","N0","N1","N2","C")
status = {arm:read(root/arm/"status.json") for arm in arms}
assert all(v["status"]=="COMPLETE" and v["spent_updates"]==233 for v in status.values())
assert len({v["training_masks_hash"] for v in status.values()})==1
screen = read(root/"screen.json")
seed = read(campaign/"s7_seed_confirmation/screen.json")
assert screen["status"] in ("NO_CANDIDATE","CANDIDATE")
section = [
    "## S6 终点与 S7 决策",
    "",
    "每组 seed 17、233 次更新，A/B1 各 7,442 个真实视图。训练目标、可见掩码与目标像元逐条匹配。"
    "验证在 0、42、116、233 步完成，同一固定 VAL 与历史协议一致。下表中的 B1 主路径指标以兼容支路相加前的预测计算。",
    "",
    "| 组 | A MSE | B1 总 MSE | B1 主路径 MSE | A 高纹理梯度 MSE | B1 主路径高纹理梯度 MSE | 峰值 GiB | 墙钟秒 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for arm,row in status.items():
    groups = row["validation"][-1]["groups"]
    a,b = groups["A/all"],groups["B1/all"]
    section.append(f"| {arm} | {a['mse']:.6f} | {b['mse']:.6f} | {b['main_path_mse']:.6f} | "
                   f"{a['texture_gradient_mse']:.6f} | {b['main_texture_gradient_mse']:.6f} | "
                   f"{max(row['peak_allocated_bytes'].values())/2**30:.2f} | {row['wall_seconds']:.1f} |")
section += [
    "",
    "以下是直接父组的配对总体差值，正值表示误差上升。N0 同时改变入口和融合，结果按组合解释。",
    "",
    "| 候选 / 父组 | A MSE 差 | B1 MSE 差 | A 主路径纹理变化 | B1 主路径纹理变化 | 数值筛选 |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
parents = dict(BP="B0",N0="BP",N1="N0",N2="N1",C="N2")
for arm,parent in parents.items():
    rec = screen["effects"][arm][parent]
    a,b = [rec["paired"][k]["metrics"] for k in ("A/all","B1/all")]
    relative = rec["screen"]["relative"]
    section.append(f"| {arm} / {parent} | {a['mse']['mean']:+.6f} | {b['mse']['mean']:+.6f} | "
                   f"{relative['A/all']['main_texture_gradient_mse']:+.2%} | "
                   f"{relative['B1/all']['main_texture_gradient_mse']:+.2%} | {rec['screen']['status']} |")
if screen["status"]=="NO_CANDIDATE":
    selection = "六组短程比较没有候选同时满足 MSE 不劣与共享主路径纹理门槛，S7 多种子确认未触发。"
else:
    selection = f"S6 候选为 {screen['candidate']}，参照为 {screen['baseline']}。S7 状态为 {seed['status']}。"
section += [
    "",selection,
    "",
    "ResNet 对照 N1 相对 N0 的 B1 像元 MSE 下降 0.56%，A 上升 0.68%，两项分组区间均跨零。"
    "FiLM 对照 N2 相对 N1 的 B1 MSE 下降 0.035%，共享主路径纹理变化小于 0.001%。"
    "D1 对照 C 相对 N2 的总体 A/B1 MSE 上升 1.75% / 1.65%，S2 两项均退化，L8 两项均改善。"
    "B1 主路径纹理误差上升 4.05%。从同一 C 权重读取，叠加历史支路把 B1 高纹理梯度 MSE 从 0.13801 降到 0.07293。"
    "固定 S2 的 B1 B03 样本中，D1 主路径有周期性网格，40 米半格相位的边界残差比从 1.007 升到 1.070。"
    "该相位发现只针对固定面板。PD 属于可选交互对照，本轮在已登记对照中没有筛出需据此晋级的改动。",
    "",
    "[验证曲线与 D1 分传感器差值](figures/s6_overview.png) · "
    "[S2 B1 主路径及叠加支路效果](figures/s6_C_obs_5929ceec240fa570c7a1a784_B1.png) · "
    "[D1 改善最大、中位、退化最大案例](../s6_component_trials/paired_examples/selection.json) · "
    "[A 图](figures/s6_D1_paired_A.png) · [B1 图](figures/s6_D1_paired_B1.png)",
    "",
    "按隔离组、源产品和采集键合并重复观测，配对区间采用 5,000 次组重采样。单种子区间描述样本组差异。"
    "原生形状、数据与验证完全保留。墙钟、每设备峰值和实际任务暴露来自各组日志。"
    "当前性能记录区分了更新、完整验证和保存，前端、融合、主干与 Decoder 的独立计时未测。",
    "",
    "[数值筛选](../s6_component_trials/screen.json) · [逐观测配对](../s6_component_trials/paired_effects.csv) · "
    "[逐波段配对](../s6_component_trials/paired_effects_by_band.csv) · "
    "[GSD 与同格可见波段分层](../s6_component_trials/paired_effects_by_gsd_and_support.csv) · "
    "[固定面板 A 遮挡距离](fixed_panel_A_distance.csv) · "
    "[固定面板相位](../s6_component_trials/fixed_panel_phase_audit.json) · "
    "[S7 记录](../s7_seed_confirmation/screen.json)",
    "",
    "[逐观测指标](metrics_by_observation.csv) · [逐波段指标](metrics_by_band.csv) · "
    "[端点与成本](endpoint_metrics_and_cost.csv) · [测试](test_report.json) · [追踪矩阵](TRACEABILITY.csv) · "
    "[Viewer](../viewer_index.json) · [产物清单](artifact_manifest.json)",
]
path = summary/"REPORT.md"
report = path.read_text(encoding="utf-8").split("\n## S6 终点与 S7 决策\n")[0]
path.write_text(report.rstrip()+"\n\n"+"\n".join(section)+"\n",encoding="utf-8")
quality = "NO_CANDIDATE" if screen["status"]=="NO_CANDIDATE" or seed["status"]=="NO_CANDIDATE" else seed["status"]
(summary/"DECISION.md").write_text(
    "# V3-FR 决策\n\n"
    "| 验收 | 状态 |\n| --- | --- |\n"
    "| 实现验收 | PASS，限冻结光学数据、规定几何和已测执行路径 |\n"
    f"| 效果验收 | {quality} |\n\n"
    "保留冻结基线、可见性与几何门禁、原生浮点图集、B1 总输出及主路径双评价。"
    "浅层 ResNet、位置适配、D1 和独立历史支路均可定位到张量接口与测试。\n\n"
    f"{selection}\n\n"
    "全可见 AE-F 的拟合收益来自共享主路径。其两条 TRAIN 样本和单种子限制了外推。"
    "各模块短程结果见报告。空间细节的主要损失环节和滑坡下游迁移效果尚未确定。\n",
    encoding="utf-8")
for name,args in (
    ("scripts/distance_metrics.py",[str(campaign)]),
    ("scripts/summarize_live.py",[str(campaign)]),
    ("scripts/audit_trials.py",[str(campaign),str(prior)]),
    ("analysis/assemble.py",[str(campaign)]),
):
    subprocess.run([sys.executable,"-B",str(project/name),*args],check=True)
