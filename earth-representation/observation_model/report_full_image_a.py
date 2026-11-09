"""Publish this bounded full-original A only; never launches another training phase."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'observation_model/outputs/token_A_full'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while not (OUT/'result.json').exists():
        if not args.watch:return
        time.sleep(20)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((OUT/'result.json').read_text());controls=json.loads((OUT/'controls.json').read_text())
    run=OUT/'tvlp/A_full_native50_v2';records=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    train=[x for x in records if x['phase']=='train'];history=r['history']
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].plot([x['global_step'] for x in train],[x['metrics']['loss'] for x in train],alpha=.6,lw=.6);axes[0].set_yscale('log');axes[0].set_ylabel('完整原图A训练MSE')
    axes[1].plot([x['step'] for x in history],[x['score'] for x in history],marker='o',label='模型固定验证')
    for key,label,color in [('spatial','同波段空间插值','black'),('spectral','其他波段线性回归','tab:orange')]:axes[1].axhline(controls['val'][key]['score'],ls='--',label=label,color=color)
    axes[1].legend();axes[1].set_ylabel('固定地域验证MSE')
    axes[2].plot([x['global_step'] for x in train],[x['metrics']['lr'] for x in train]);axes[2].set_ylabel('实际学习率')
    for ax in axes:ax.set_xlabel('更新步数（每步遍历100个原始观测）');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=150);plt.close(fig)
    exe='C:/Users/yzh/Downloads/training_viewer_v1.1_spec_and_examples/training_viewer_v1/.venv/Scripts/training-viewer.exe'
    validation=json.loads(subprocess.run([exe,'validate',str(run),'--json'],capture_output=True,text=True,encoding='utf-8',check=True).stdout)
    assert validation['error_count']==0 and validation['warning_count']==0
    points={(x['session_id'],x['global_step']) for x in train}
    lines=['## 完整原图实验A结果','',
        f"本轮完成{r['step']}次更新，停止原因为`{r['stop_reason']}`，最佳权重在第{r['best_step']}步。每次更新遍历100个训练观测；验证使用全部49份S2、47份Landsat原始留出文件。全部输入为完整原生文件范围，不是240m裁剪，不是插值放大。",'',
        '| 方法 | 训练MSE | 固定地域验证MSE |','|---|---:|---:|']
    for key,label in [('spatial','同波段空间插值'),('spectral','同位置其他波段线性回归')]:lines.append(f"| {label} | {controls['train'][key]['score']:.6f} | {controls['val'][key]['score']:.6f} |")
    lines += [f"| 模型最佳 | {r['best_model']['train']:.6f} | {r['best_model']['val']:.6f} |",f"| 模型末态 | — | {r['last_model']['val']:.6f} |",'',
        '![完整原图A训练与验证](outputs/token_A_full/training_curves.png)','',
        '左图是实际逐步训练损失，中图是固定地域验证及同目标简单参照，右图是实际学习率。单个60m缺口在Landsat仍只有2×2个真实像元；完整输入约49×49，二者必须区分。影像文件来自已下载的原生产品窗口，不称整幅卫星大景。','',
        f"主线程日志共记录{len(points)}次真实更新，回滚重算包含在内。验证、归一化、分波段结果、实际原文件范围及源文件哈希见`outputs/token_A_full/`。此次只完成A，不自动启动C、跨源或物理约束训练。"]
    path=ROOT/'observation_model/README.md';text=path.read_text();start='<!-- FULL_A_RESULTS_START -->';end='<!-- FULL_A_RESULTS_END -->'
    if start in text:text=text[:text.index(start)]+text[text.index(end)+len(end):]
    path.write_text(text.rstrip()+'\n\n'+start+'\n'+'\n'.join(lines)+'\n'+end+'\n')
    (OUT/'delivery.json').write_text(json.dumps(dict(status='FULL_IMAGE_A_COMPLETE_NO_DOWNSTREAM_TRAINING',step=r['step'],logged_updates=len(points),validation=validation,checkpoint_hashes={x:hashlib.sha256((OUT/x).read_bytes()).hexdigest() for x in ['best.pt','last.pt']},scope='A only'),indent=2))
    print('Full-original A report published',flush=True)


if __name__=='__main__':main()
