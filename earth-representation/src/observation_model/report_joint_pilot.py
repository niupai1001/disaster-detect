"""Publish the bounded pilot's actual VAL comparison to the stable entry."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'observation_model/outputs/joint_v3'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while not (OUT/'pilot_comparison.json').exists():
        status=json.loads((OUT/'pilot_status.json').read_text())
        if status['status']=='failed':raise RuntimeError('pilot failed; report not generated')
        if not args.watch:return
        time.sleep(15)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((OUT/'pilot_comparison.json').read_text());values=result['results']
    training=json.loads((OUT/'pilot1000/result.json').read_text())
    exe='C:/Users/yzh/Downloads/training_viewer_v1.1_spec_and_examples/training_viewer_v1/.venv/Scripts/training-viewer.exe'
    validation=json.loads(subprocess.run([exe,'validate',str(OUT/'tvlp/pilot1000'),'--json'],capture_output=True,text=True,encoding='utf-8',check=True).stdout)
    assert validation['error_count']==0 and validation['warning_count']==0
    records=[json.loads(line) for line in (OUT/'tvlp/pilot1000/metrics.jsonl').read_text().splitlines()]
    actual={(r['session_id'],r['global_step']) for r in records if r['phase']=='train'}
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,4));history=training['validation_history']
    train=[r for r in records if r['phase']=='train'];axes[0].plot([r['global_step'] for r in train],[r['metrics']['loss'] for r in train],alpha=.4,lw=.6)
    axes[0].set_xlabel('更新步数');axes[0].set_ylabel('联合归一化MSE');axes[0].set_title('随机训练任务损失：不直接用于排名')
    axes[1].plot([r['step'] for r in history],[r['singleton_nrmse'] for r in history],marker='o',label='新候选固定验证')
    for k,label,color in [('ridge','Ridge','black'),('old_best','旧最佳模型','tab:orange')]:axes[1].axhline(values[k]['singleton_nrmse'],ls='--',color=color,label=label)
    axes[1].set_xlabel('更新步数');axes[1].set_ylabel('固定单隐藏nRMSE');axes[1].set_title('相同地域、全部波段与完整目标网格');axes[1].legend()
    for a in axes:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'pilot_validation.png',dpi=150);plt.close(fig)
    lines=['## 联合监督1000步固定地域验证结果','',
        '以下均为同一冻结VAL成员；综合评分是单隐藏与固定多隐藏归一化MSE的等权平均。主任务TEST未打开，1000步上限不代表充分收敛。','',
        '| 方法 | 综合验证MSE | 单隐藏nRMSE | 固定多隐藏MSE |','|---|---:|---:|---:|']
    for k,label in [('new_best','联合监督新最佳'),('new_last','联合监督末态'),('old_best','旧Transformer最佳'),('ridge','条件Ridge')]:
        v=values[k];lines.append(f"| {label} | {v['score']:.6f} | {v['singleton_nrmse']:.6f} | {v['multiple_mse']:.6f} |")
    lines += ['', '![联合监督固定地域验证](outputs/joint_v3/pilot_validation.png)','',
        '左图是每次更新的随机联合训练损失，右图是固定单隐藏验证与同版本参照。综合选模还包含固定合法多隐藏任务，不能只按右图选权重。', '',
        '候选组合改变了任务采样、目标尺度、优化日程和参数分组，且关闭增强；旧权重训练更久，不是严格等预算或单因素消融。小样本拟合收益不能代替此处验证；跨源训练和冻结表征收益尚未由本轮证明。', '',
        f"本轮实际日志记录{len(actual)}次优化器更新。TVLP校验0错误、0警告；证据见`outputs/joint_v3/pilot_delivery.json`，原始比较见`outputs/joint_v3/pilot_comparison.json`。"]
    path=ROOT/'observation_model/README.md';text=path.read_text();start='<!-- JOINT_PILOT_RESULTS_START -->';end='<!-- JOINT_PILOT_RESULTS_END -->'
    if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
    path.write_text(text.rstrip()+'\n\n'+start+'\n'+'\n'.join(lines)+'\n'+end+'\n')
    (OUT/'pilot_delivery.json').write_text(json.dumps(dict(status='PILOT_REPORTED_NOT_FULL_PLAN_COMPLETE',logged_updates=len(actual),viewer_validation=validation,
        checkpoint_hashes={n:hashlib.sha256((OUT/'pilot1000'/n).read_bytes()).hexdigest() for n in ['best.pt','last.pt']},comparison='pilot_comparison.json',remaining='review actual VAL result before allocating further training; old queue remains paused'),indent=2))
    print('Pilot VAL results published to stable README',flush=True)


if __name__=='__main__':main()
