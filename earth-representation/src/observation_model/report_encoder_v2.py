"""Publish the bounded G2 evidence; never starts G3 or model optimization."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .check_encoder_v2 import DEST,G1
from .expansion_data import ROOT,OUT,read,save,digest


def main():
    required=['shape_tests.json','geometry_tests.json','hypernet_tests.json','leakage_tests.json',
        'cpu_acceptance.json','decoder_context_tests.json','memory_profile.json']
    assert all(read(DEST/p)['status']=='PASS' for p in required)
    old_policy=read(OUT/'release.json')['old_source_policy_sha256']
    assert digest(ROOT/'observation_model/execution_policy.json')==old_policy
    c=read(DEST/'config.json');h=read(DEST/'hypernet_tests.json');g=read(DEST/'memory_profile.json')
    leak=read(DEST/'leakage_tests.json');cpu=read(DEST/'cpu_acceptance.json')
    assert cpu['model_hash_before']==g['model_hash_before']==g['model_hash_after']
    names=['encoder_v2.py','check_encoder_v2.py','check_decoder_context.py','report_encoder_v2.py',
        'model.py','expanded_dataset.py']
    sources={}
    for name in names:
        p=ROOT/'observation_model'/name;t=DEST/'source'/name;t.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,t);sources[name]=digest(p)
    save(DEST/'source_manifest.json',dict(files=sources,aggregate_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
        GPU_executed_source_manifest='gpu_profile_source_manifest.json',
        post_GPU_change='Global-H fallback only; CPU-tested with native path bitwise equivalence; not GPU-profiled'))
    save(DEST/'approval_state.json',dict(stage='G2',status='PASS',authorized_by='User: G1 acceptance then G2',
        optimizer_updates=0,formal_training_authorized=False,G3_started=False,G4_started=False,
        GPU_forward_backward_runs=1,global_H_fallback_CPU_verified=True,global_H_fallback_GPU_verified=False,
        data_manifest_sha256=read(OUT/'release.json')['manifest_sha256'],
        G1_contract_sha256=read(G1/'data_contract_tests.json')['observation_contracts_sha256'],
        old_execution_policy_sha256=old_policy,next_gate='G3 task curriculum/Viewer engineering, not started'))
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(16,5),layout='constrained');ax.set_xlim(0,16);ax.set_ylim(0,5);ax.axis('off')
    boxes=[(.2,1.6,2.1,1.4,'原生光学观测\n10 / 20 / 30 m\n逐波段网格与QA'),
        (2.9,1.6,2.1,1.4,'可见像元隔离\n局部空间矩\n隐藏值不参与运算'),
        (5.6,1.6,2.1,1.4,'共享条件投影\nW = W0 + U diag(h) V\n低秩 rank=8'),
        (8.3,1.6,2.1,1.4,'实际面积支持\n波段集合交叉注意力\n可变地理Token'),
        (11.,1.6,2.1,1.4,'4层局部空间注意力\nD=128，窗口8×8\n独立输出 H'),
        (13.7,1.6,2.1,1.4,'目标条件解码器\n仅读取 H 与目标元数据\n回到目标原生网格')]
    for x,y,w,height,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,height,boxstyle='round,pad=.08',facecolor='#edf4fb',edgecolor='#245b88'))
        ax.text(x+w/2,y+height/2,label,ha='center',va='center',fontsize=10)
    for i in range(len(boxes)-1):
        x,y,w,height,_=boxes[i];xx=boxes[i+1][0]
        ax.annotate('',xy=(xx-.1,y+height/2),xytext=(x+w+.1,y+height/2),arrowprops=dict(arrowstyle='->',lw=1.6))
    ax.text(6.65,4,'SRF固定谱基积分 + log(gsd_x, gsd_y)\n真实生成投影权重；不是传感器类别ID',ha='center',va='center',fontsize=11)
    ax.annotate('',xy=(6.65,3.1),xytext=(6.65,3.6),arrowprops=dict(arrowstyle='->',lw=1.5))
    ax.text(8,.6,'线性投影与面积积分可交换，非线性在积分之后；不创建全分辨率 D 通道图。\n大洞缺少局部可见Token时，仅用 H 的全局均值补位。无原始像元旁路；0次优化更新。',ha='center',fontsize=11)
    ax.set_title('Encoder V2 · G2实际实现与连接',fontsize=15)
    fig.savefig(DEST/'architecture.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    bars=axes[0].bar(['隐藏值+QA','整波段移除','波段顺序置换'],[leak['A_prediction_intervention_max'],0,leak['permutation_max_abs']],color=['#4285b4','#4285b4','#e7a141'])
    axes[0].bar_label(bars,fmt='%.2g',padding=3);axes[0].set_ylabel('最大绝对输出差异');axes[0].set_title('输入隔离与集合顺序检查')
    bars=axes[1].bar(['同中心、不同带宽','10m改为30m'],[h['same_center_different_bandwidth_weight_max'],h['changed_GSD_weight_max']],color='#4285b4')
    axes[1].bar_label(bars,fmt='%.2g',padding=3);axes[1].set_ylabel('实际投影权重最大差异');axes[1].set_title('物理条件确实进入权重生成')
    fig.savefig(DEST/'engineering_controls.png',dpi=150);plt.close(fig)
    text=f'''# G2：共享物理条件Encoder工程验收

## 阅读约定与设置

本关只验证可运行、可对齐、无隐藏信息泄漏，以及物理条件和真实像元确实进入数值路径。不是训练收益评价。使用G1冻结数据的两个TRAIN观测完整原生窗口：S2十波段10/20米，Landsat8七波段30米；具体ID与缓存哈希见members.json。没有裁成旧240米试验窗口，没有读取TEST目标，没有监督标签。CPU另做256×256、0.5米合成观测；不代表WorldView实测数据。Landsat9接口沿用同一物理描述，本关未单独做真实L9前后向，不声称其独立验证收益。

初始化为seed=0随机权重，全部FP32。没有加载历史预训练checkpoint，没有创建优化器，所有工程前后向的优化更新数均为0。固定容差在config.json预注册：置换比较atol/rtol各1e-5，隐藏值干预要求逐位不变，敏感性要求大于1e-8。

## 主表

| 检查对象 | 实际结果 | 能说明与不能说明什么 |
|---|---|---|
| 原生几何与变长输入 | 10/20/30米真实目标输出形状正确；15米错相面积均值符合手算；混合batch与单独编码一致，padding无效 | 没有先统一原始影像尺寸；跨CRS与旋转栅格显式拒绝，不暗中猜测配准 |
| 条件投影与集合融合 | 同中心不同带宽权重最大变化{h['same_center_different_bandwidth_weight_max']:.6g}；GSD变化{h['changed_GSD_weight_max']:.6g}；波段置换输出差{leak['permutation_max_abs']:.6g} | 条件进入真实投影，输出在预设浮点容差内近似顺序不变；不是SRF训练增益 |
| 遮盖与信息路径 | 空间隐藏值和QA干预差0；整波段数值/QA/SRF干预差0；隐藏输入梯度0；可见像元扰动可传入H | 未发现已测A/B隔离路径泄漏；G3随机遮盖分布、损失与课程仍未实现 |
| 一次GPU混合前后向 | forward {g['forward_seconds']:.3f}s，backward {g['backward_seconds']:.3f}s；峰值分配{g['peak_allocated_bytes']/1024**3:.3f}GiB；参数SHA不变 | RTX5090上正常原生路径可运行；不是有效batch32吞吐，也不是正式预训练耗时 |

## 架构与解释

![实际模块连接](architecture.png)

从左到右是实际数值路径；上方物理条件只影响投影权重。Decoder只接受H和目标SRF/网格，不接受目标值或原始影像。面积积分与投影的线性部分交换顺序以减少显存，非线性放在积分之后，不能将它误称与先逐像元非线性完全等价。

复用既有model.py的AttentionBlock、CrossBlock及PyTorch MultiheadAttention；没有新装依赖。没有复用DOFA完整固定Patch卷积核，因为该核尺寸不适用于连续变化的0.5/10/20/30米像元支持。新实现是共享10项局部空间矩上的128维低秩投影，rank8；条件为64个固定谱段SRF积分和两个GSD对数。像元坐标来自真实affine，先以float64减世界原点，再用float32局部运算；footprint保留float64。产品、传感器、波段字符串只用于查元数据和移除目标，不作为可学习类别ID。

每个原生像元按与地理Token的实际矩形相交面积参与聚合；有效区域先剔除，无效零值不是有效观测。逐行分块累加空间矩，避免为每个原生像元保存128通道。原生反射率没有被插值成一幅所谓共同高分辨率观测。集合融合后只保留H，没有原始值跳接或额外Fine/Context输出。

![工程控制](engineering_controls.png)

左图比较隐藏值/QA干预、整波段移除以及波段顺序置换；前两项为0，置换差仅为浮点归约误差。右图比较相同中心、不同带宽及不同GSD时实际生成的权重；非零变化证明条件生效，不证明它能改善学习。

同容量的condition_mode='off'与'frozen'已实现：关闭条件时生成权重严格等于共享权重；冻结模式只冻结超网络。三种模式保留相同模块形状与总参数规模，尚未训练比较。

## Token数量、容量与范围

固定N_ref=400、A_ref=1200²平方米、g_ref=10米、alpha=1、Nmax=8192。二维整数化采用按长宽比取最接近整数的网格，并均匀覆盖完整声明范围，不截掉图像尾部。

| 1200m×1200m下最细实际输入 | 公式目标总Token | 实际二维Token | 备注 |
|---|---:|---:|---|
| 10米 | 400 | 20×20=400 | 真实输入支持 |
| 20米 | 200 | 14×14=196 | 记录整数化差-4 |
| 30米 | 133 | 12×11=132 | 记录整数化差-1 |
| 0.5米 | 8000 | 89×90=8010 | 仅公式；未跑完整1200米合成前后向 |

实际GPU两个完整窗口得到361/196个Token，batch padding到361；这两个窗口的地面范围并非都等于1200米。10km方形0.5米保险丝只做几何预算计算，明确标记capped并限制至8190个Token，没有分配该尺寸的像元。合成0.5米前向实际只覆盖128米方形，90个Token；其局部扰动可传递，不代表细节可以精确恢复。未引入local feature bank；是否需要它仍须后续信息恢复诊断决定。

模型总参数{h['parameters']:,}，其中Encoder {h['encoder_parameters']:,}、Decoder {h['decoder_parameters']:,}。主干为四层局部/移位注意力，宽128、4头、FFN512；Decoder两层目标CrossBlock，查询分块256。不存在0.5米全局N²注意力。

## 限制与下一步

在最终审查中补齐大洞路径：若3×3局部邻域没有可见H，Decoder用已有有效H的全局均值作为一条上下文键，不读取隐藏像元或QA。600米大洞CPU检查通过，可预测且隐藏梯度为0；常规原生路径的CPU输出在补齐前后逐位一致。该补位不是更大主干，也没有训练收益证据。

GPU只创建一次必要上下文，测试执行源码保留在source_gpu_profile/。这次GPU测试在大洞补位加入前完成；补位分支只做CPU前后向，未另开GPU上下文，也未测该分支GPU耗时。memory_profile.json只对应已测常规原生路径，不能将其推广到所有遮盖、任意面积或0.5米完整窗口。

G2通过表示数值路径成立，不表示重建已学会、细节已保真或跨卫星有效。下一关G3需要任务mask、均衡损失、课程日志与Viewer门禁，目前未开始；G4/G5仍没有优化更新授权。剩余下载队列保持原状，不修改本轮成员或旧训练禁令。

## 附录：复现与产物

在项目根依次运行：

```powershell
../landslide-seg/.venv-use-independent/Scripts/python.exe -X utf8 -m observation_model.check_encoder_v2
../landslide-seg/.venv-use-independent/Scripts/python.exe -X utf8 -m observation_model.check_encoder_v2 --gpu cuda:1
../landslide-seg/.venv-use-independent/Scripts/python.exe -X utf8 -m observation_model.check_decoder_context
```

入口拒绝覆盖既有验收；以上是已执行命令说明，不应对已完成输出重复运行。GPU执行时源码与最终源码的差异及CPU等价证明见gpu_profile_source_manifest.json和decoder_context_tests.json。首轮CPU计时{cpu['wall_seconds']:.3f}秒，GPU阶段含初始化/读取的wall时间{g['wall_seconds']:.3f}秒；额外审阅、补位CPU检查与绘图不混作训练成本。云GPU收费未发生。

initialization_untrained.pt仅为随机初始化工程快照，没有best或训练末态。模型张量SHA256为`{cpu['model_hash_before']}`，CPU与GPU前后均不变。配置、成员、源码、G1输入哈希、形状/几何/条件/泄漏/梯度/显存证据在本目录固定文件中，最终聚合见source_manifest.json与delivery_hashes.json。共享研究台账仍由其他聊天认领，本轮不越过锁改写；稳定研究入口仍为observation_model/README.md。
'''
    (DEST/'architecture.md').write_text(text,encoding='utf-8')
    (DEST/'model_card.md').write_text('# Encoder V2 · G2工程模型\n\n随机初始化、无训练，详见[唯一架构与验收报告](architecture.md)。\n\nencode返回features、footprints、valid、scale_m、source_support和grids；predict只从H及目标物理元数据生成原生网格估计。不能作为已训练权重交付，不能报告语义、跨卫星或0.5米实测收益。\n',encoding='utf-8')
    p=ROOT/'observation_model/README.md';body=p.read_text(encoding='utf-8')
    start=body.index('> 当前工作：');end=body.index('\n',start)
    body=body[:start]+'> 当前工作：G2架构工程验收通过；物理条件投影、面积支持融合、独立Encoder与原生目标Decoder已实现。0优化更新，未进入G3或恢复旧训练。'+body[end:]
    body=body.replace('G2尚未授权，未启动架构工程、G4诊断或正式训练。','G1完成时未启动架构工程；后续G2授权与实际验收见下方新入口，G4诊断和正式训练仍未启动。')
    block='\n<!-- ENCODER_V2_G2_START -->\n## Encoder V2：G2核心架构验收\n\n按用户追加授权完成[架构与工程验收](outputs/encoder_v2/G2_arch/architecture.md)：共享SRF/GSD条件低秩投影、实际面积支持聚合、波段集合融合、四层局部空间主干及只读H的目标解码器。真实10/20/30米、合成0.5米、波段乱序、条件生效、隐藏值/QA干预与有限梯度检查通过。一次GPU正常原生路径前后向约1.41/0.25秒，峰值分配约0.94GiB；后补大洞上下文分支仅CPU验证，GPU原生路径前后CPU输出逐位相同，执行源码分别留存。\n\n[批准状态](outputs/encoder_v2/G2_arch/approval_state.json)、[源码清单](outputs/encoder_v2/G2_arch/source_manifest.json)、[显存报告](outputs/encoder_v2/G2_arch/memory_profile.json)和[大洞检查](outputs/encoder_v2/G2_arch/decoder_context_tests.json)为事实入口。总参数1423129，随机初始化，优化更新0；G2不代表已学会重建。G3任务/Viewer尚未启动，下载队列、冻结数据及旧训练禁令保持。\n<!-- ENCODER_V2_G2_END -->\n'
    if '<!-- ENCODER_V2_G2_START -->' in body:
        before,rest=body.split('<!-- ENCODER_V2_G2_START -->',1)
        _,after=rest.split('<!-- ENCODER_V2_G2_END -->',1)
        body=before.rstrip()+block+after
    else:body+=block
    p.write_text(body,encoding='utf-8')
    save(DEST/'delivery_hashes.json',{p.name:digest(p) for p in DEST.iterdir() if p.is_file() and p.name!='delivery_hashes.json'})
    print('G2_DELIVERED_NO_TRAINING')


if __name__=='__main__':main()
