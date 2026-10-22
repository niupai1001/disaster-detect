"""Read saved results, verify completed runs, and render the stage decision report."""
import json
import hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'stage_review_2026-10-20'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 11})
def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))
p = read('reports/p13/results_summary.json')
q = read('reports/p2m1/capacity_summary.json')
keys = ['P13-FewEvents', 'P13-AllEvents', 'B0-AllTrain-ctl', 'B2-AllTrain']
runs = [p[keys[0]], p[keys[1]], q[keys[2]], q[keys[3]]]
names = ['轻量主干＋末端细节／三事件', '轻量主干＋末端细节／九事件', '轻量主干＋末端细节／全量', '较大主干＋末端细节／全量']
colors = ['#bf744b', '#277d9a', '#277d9a', '#bf744b']
evidence = {}
hist = []
for k, r in zip(keys, runs):
    d = ROOT / r['run_dir']
    s = json.loads((d / 'train_summary.json').read_text(encoding='utf-8'))
    assert s['complete'] and s['global_step'] == 3600 and s['seed'] == 1234
    assert (d / 'best.pt').is_file() and (d / 'last.pt').is_file()
    ev = json.loads((d / 'eval/eval_best_val.json').read_text(encoding='utf-8'))
    assert ev['n_samples'] == 669
    assert abs(ev['event_mean_iou'] - r['best_val']['event_mean_iou']) < 1e-12
    hist.append(s['history'])
    evidence[k] = {'run_dir': r['run_dir'], 'complete': True, 'global_step': 3600,
                   'summary_sha256': hashlib.sha256((d / 'train_summary.json').read_bytes()).hexdigest(),
                   'best_eval_sha256': hashlib.sha256((d / 'eval/eval_best_val.json').read_bytes()).hexdigest()}
(OUT / 'verification.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding='utf-8')
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, ix, title in zip(axes, [(0,1), (2,3)], ['同N=1251：改变事件组成', '同N=2223：改变模型规格']):
    labels = ['三事件', '九事件'] if ix[0] == 0 else ['轻量主干', '较大主干']
    for j, i in enumerate(ix):
        vals = [runs[i]['best_val']['by_event'][e]['iou'] for e in ['hiroshima','thrissur']]
        ax.bar([x + (j-.5)*.34 for x in range(2)], vals, width=.34, label=labels[j], color=colors[i])
        for x,v in enumerate(vals):
            ax.text(x+(j-.5)*.34, v+.005, f'{v:.3f}', ha='center', fontsize=10)
    ax.set_xticks([0,1], ['Hiroshima', 'Thrissur'])
    ax.set_title(title); ax.set_ylim(0,.37); ax.legend(loc='lower center')
    ax.set_ylabel('DEV前景IoU（阈值0.5）'); ax.grid(axis='y', alpha=.2)
fig.tight_layout(); fig.savefig(OUT/'event_comparison.png', dpi=170); plt.close(fig)
fig, axes = plt.subplots(1,2,figsize=(13,4.5))
for ax, ix, title in zip(axes, [(0,1),(2,3)], ['事件组成：九事件收益仍伴随后期退化', '模型规格：训练拟合更高，DEV更低']):
    for i in ix:
        h = hist[i]; xs = [x['global_step'] for x in h]
        label = ['三事件','九事件','轻量主干','较大主干'][i]
        ax.plot(xs,[x['dev']['event_mean_iou'] for x in h],color=colors[i],label=label+' DEV')
        ax.plot(xs,[x['train_clean']['pooled']['iou'] for x in h],color=colors[i],ls='--',label=label+' TRAIN')
    ax.set(title=title,xlabel='成功更新步数',ylabel='IoU'); ax.legend(fontsize=9); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(OUT/'training_curves.png',dpi=170); plt.close(fig)
lines = ['# 事件覆盖与模型规格：结果复核及阶段决策（2026-10-20）',
'', '## 阅读约定与实验设置', '',
'本次读取既有真实训练、checkpoint存在性和独立评测产物，重新生成图表并核验摘要一致性；未重训、未重新运行GPU推理。四组均已完成3600次成功更新，seed1234、FP32、有效batch64、230400次曝光。输入为单期Sentinel-2十波段128×128，采用原BCE＋Dice、valid mask、增强和调度。每100步按两个DEV事件等权平均IoU选择best，阈值0.5。',
'', 'DEV固定669块（Hiroshima448、Thrissur221）。事件平均IoU先在事件内汇总TP/FP/FN，再对事件等权平均；合并IoU、Dice、Precision、Recall、AP使用全部有效像素。对象召回使用可评对象独立一对一匹配、8邻域、IoU门槛0.25，分母1679；边界F1为1像素容差。',
'', '两项对照分别解释：同N事件组成组共享703块交集拟合的统计；全量规格组共享原训练统计。不得将九事件1251与全量2223的差值解释成纯扩量效应。没有重新评测Italy或New Zealand，不宣称历史盲测。',
'', '## 主表（合并）', '',
'| 架构与训练组成 | TRAIN图块 | DEV事件平均IoU | 合并IoU | Dice | Precision | Recall | AP | 对象召回 | 边界F1 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for n,r in zip(names,runs):
    b=r['best_val']; vals=[b[x] for x in ['event_mean_iou','iou','dice','precision','recall','ap']]+[b['objects']['object_recall_evaluable'],b['boundary_f1']]
    lines.append('| '+n+f" | {r['n_train']} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
lines += ['', '## 图与解释', '', '![逐事件比较](stage_review_2026-10-20/event_comparison.png)', '',
'左列为同N三事件与九事件，右列为同数据轻量与较大模型；横轴为DEV事件，纵轴为前景IoU。应看左列收益高度不均：Hiroshima约+0.1106，Thrissur约+0.0108；右列较大模型在两事件均下降。图中没有跨协议排名。', '',
'![实际训练曲线](stage_review_2026-10-20/training_curves.png)', '',
'左列为事件组成，右列为模型规格；实线是DEV事件平均IoU，虚线是TRAIN合并IoU，两者聚合口径不同，不应把垂直距离解释成严格同口径泛化差值。应看随更新步数增加，训练拟合继续提高而DEV整体退化；曲线全部来自逐检查点原始history，并非插造。', '',
'## 展开结论', '',
'**优先推进训练事件覆盖与跨事件学习。** 固定1251独立图块，九事件的DEV平均IoU从0.2300升至0.2908（+0.0607），AP从0.2653升至0.4100，对象命中从607增至879，边界F1从0.3305升至0.4562。合理但未验证的机制是更丰富的地理外观减少对少数训练域的依赖；当前证据直接支持的是这两种具体事件组成的差别。收益主要来自Hiroshima，且三事件组合仍有小对象比例高约0.101的残余差异，不能归纳为每增加事件都有效，也不能将0.0607外推为再收集六个事件的预期收益。', '',
'**当前固定预算不支持升级较大同族模型。** 同版本轻量控制组为0.3082，较大组0.2698（−0.0384）；AP与对象召回同步下降，Precision提高伴随Recall从0.4777降至0.3728。较大组末期TRAIN IoU约0.5616，高于轻量组0.4894，但DEV降至0.1723；这支持训练拟合改善未转化为留出收益，过拟合或域依赖是未验证的机制解释。规格比较同时改变encoder、decoder维度及参数量，固定预算负结果不能证明其充分优化后的能力上限，也不支持否定遥感预训练。', '',
'**事件覆盖没有解决小对象与后期退化。** 九事件组1–4像素对象仅命中13/77；更多事件改善中小尺度总体检出，但不能据此宣布十米单期输入已足够表达极小目标。两组事件对照best在200/400步，末期DEV分别降至0.1421/0.2040，仍需研究可迁移表征。此观察不证明信息不足，应保留官方条件单期及多时相/DEM参考诊断。', '',
'## 限制与下一步', '',
'单seed不宣称统计显著。已有按事件分层的图块bootstrap区间（事件组成best差值约[+0.0422,+0.0778]）仅描述固定模型、固定两个DEV事件内图块重采样的不确定性；没有覆盖训练随机性、事件组合选择或新事件泛化，空间相关性也可能使区间偏窄。best使用同一DEV选择，last方向一致只能增加描述性证据。', '',
'权重加载记录支持本地B2张量完整加载，并有文件SHA256，但上游身份核验尚未闭环。特别是decoder_hidden_size=768是特征维度，不等于预训练图像尺寸768；既有接口报告将512目录名与decoder维度视为命名冲突的判断不成立。保留原文，仅在本报告纠正解释；不能据此确认上游权重身份，也不能把已训练结果称为随机初始化。', '',
'下一阶段按原决策进入P3-B，保留轻量LateDetail参照，不追加loss、学习率或交换模块微调。下一批至多三项：①冻结另一组仅依据TRAIN属性的同N少事件对照，检验当前收益是否依赖选定组合；②在同一模型与预算下比较原采样和事件均衡采样，区分覆盖与曝光分配，先记录小事件重复曝光；③只围绕候选新事件做合法来源、标签和留出冲突检查，通过后再定义扩充实验。以上尚未执行，不恢复三seed。官方条件单期对照及输入充分性诊断仍为未完成项，多时相/DEM只作独立参考。', '',
'## 附录（逐seed、成本、产物索引）', '',
'全部为seed1234。成本来自本次同版本运行；时长受运行环境影响，非严格速度基准。', '',
'| 架构与训练组成 | best步 | last DEV IoU | 参数M | 秒 | 峰值显存MiB |',
'|---|---:|---:|---:|---:|---:|']
for n,r in zip(names,runs):
    lines.append(f"| {n} | {r['best_step']} | {r['last_val']['event_mean_iou']:.4f} | {r['params']['total']/1e6:.2f} | {r['seconds']:.0f} | {r['peak_alloc_mib']:.0f} |")
lines += ['', '复核脚本：[report_stage_progression_review.py](../scripts/report_stage_progression_review.py)。复核哈希：[verification.json](stage_review_2026-10-20/verification.json)。训练配置位于`configs/segformer/p13/`与`configs/segformer/p2m1/`；各run目录保存resolved_config、run_meta、train_summary、metrics、sample_exposures及best/last。', '',
'命令（项目根目录PowerShell）：', '', '```powershell', '& C:/Users/yzh/.workbuddy/binaries/python/envs/tm/Scripts/python.exe scripts/report_stage_progression_review.py', '```', '',
'详细组成与选样规则：[composition.md](../reports/p13/composition.md)；完整历史汇总：[stage_report](../reports/p13_p2m1_stage_report_2026-10-20.md)；接口与本地权重记录：[interface](../reports/p2m1/b2_interface_validation.md)。', '', '四组原始训练目录：', '']
lines += ['- `'+r['run_dir']+'`' for r in runs]
(ROOT/'docs/stage_results_review_2026-10-20.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Verified 4 completed runs; wrote report, 2 figures, and summary/evaluation hashes.')

