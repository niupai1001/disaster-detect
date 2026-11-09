"""Finalize G3 evidence and prepare the separately authorized bounded G4 gate."""
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from .encoder_v2_runtime import BASE,G3,G4,LOG_ROOT,VIEWER,VIEWER_DATA
from .expansion_data import ROOT,OUT,read,save,digest


def main():
    names=['loss_tests.json','scheduler_tests.json','resume_tests.json','dry_run.json','leakage_tests.json',
        'accumulation_loss_test.json','control_unit_tests.json']
    assert all(read(G3/n)['status']=='PASS' for n in names)
    assert (G3/'cases/index.json').exists()
    command=[str(VIEWER),'validate',str(LOG_ROOT/'encoder_v2_G3_zero_update'),'--json','--data-dir',str(VIEWER_DATA)]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',check=True)
    validation=json.loads(result.stdout);assert validation['error_count']==validation['warning_count']==0
    with urllib.request.urlopen('http://127.0.0.1:8766/api/v1/health',timeout=5) as response:health=json.load(response)
    assert health['status']=='ok'
    validation.update(status='PASS',url='http://127.0.0.1:8766/',attached_before_dry_run=True,
        viewer_attached_before_first_update=False,task_card=(LOG_ROOT/'encoder_v2_G3_zero_update/artifacts/task_card.html').is_file(),health=health)
    assert validation['task_card'];save(G3/'viewer_preflight.json',validation)
    assert digest(ROOT/'observation_model/execution_policy.json')==read(BASE/'G2_arch/approval_state.json')['old_execution_policy_sha256']
    dry=read(G3/'dry_run.json');assert dry['model_hash_before']==dry['model_hash_after']
    save(G3/'capacity_scope.json',dict(scored_views=8,preview_encoder_calls=12,gradient_encoder_calls=2,
        GPU_total_encoder_calls_derived_from_executed_control_flow=22,
        note='Original capacity/encoder_views records scored views only; preview and gradient diagnostics have additional calls. CPU supplement checks/cases are separate; no optimizer updates.'))
    sources=['encoder_v2.py','encoder_v2_tasks.py','encoder_v2_runtime.py','g3_tasks.py','check_g3_tasks.py','g3_cases.py','accept_g3.py']
    hashes={}
    for name in sources:
        p=ROOT/'observation_model'/name;t=G3/'source_final'/name;t.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,t);hashes[name]=digest(p)
    save(G3/'final_source_manifest.json',hashes)
    save(G3/'acceptance.json',dict(status='PASS',stage='G3',optimizer_updates=0,
        data_manifest_sha256=read(OUT/'release.json')['manifest_sha256'],model_hash=dry['model_hash_after'],
        viewer_errors=0,viewer_warnings=0,task_card_verified=True,source_execution='source_manifest.json',
        source_preview_and_supplements='final_source_manifest.json',nonempty_optimizer_restore='not claimed; planned at G4 update50',
        next_gate='Separate G4 authorization from user-configured automation90; no G5 authorization'))
    report='''# G3：真实任务、课程与Viewer验收

G3验收通过，优化更新严格0。使用G1冻结数据及G2随机初始化，原生网格与权重未改变。八份成员为四TRAIN、四地理隔离VAL；GPU任务预检实际使用每角色/传感器首份共四份，其他成员供有界诊断。不是完整训练池评估，不据此作泛化结论。

| 内容 | 实际证据 | 边界 |
|---|---|---|
| 空间任务A | 60米世界坐标Patch，候选60%面积；每成员两套随机多位置计划，固定验证可复现 | Patch锚定真实较粗网格；任何相交原生像元删除，边缘与QA无效像元不监督，实际像元比例另记 |
| 整波段任务B | 目标波段从输入数值与QA删除；监督整个原生有效目标 | A/B混合时B目标也从A输入及A目标列表移除；两个分支独立encode |
| 目标与课程 | 损失手算2.375通过；像元数不改变传感器/波段/观测权重；课程五端点与连续性通过 | 仅候选课程，尚非正式预训练策略；G3实际没有学习率更新 |
| 无泄漏与恢复 | 实际A/B任务隐藏值+10000及隐藏QA改动不影响可见编码；主线程心跳与新会话恢复通过 | Adam状态为空，不能宣称真实训练恢复已通过；G4第50步另验非空状态 |
| Viewer | TVLP1.0校验0错误、0警告；task_card及全部预览存在 | 随机初始化诊断/验证指标是0步前向，不是训练曲线 |

![空间遮盖计划一](preview/A_TRAIN_0.png)

每行依次是S2原生10米、20米及Landsat30米。六列为原始观测、QA、可见像元、真实监督区、原图叠加Token网格用于对齐、原生目标网格。网格背景用于几何展示，不表示隐藏真值进入了Encoder。网线稀疏展示以保持可读，实际目标像元数见标题。

![空间遮盖计划二](preview/A_TRAIN_1.png)

同一批真实观测的另一随机计划；应看遮盖位置确实变化，且每次是大量Patch而非固定一个小洞。地面范围使用完整本地原生窗口，不回到240米裁剪。

![整波段任务](preview/B_TRAIN.png)

目标波段整条缺失；另列同位置仍可见波段。目标QA只进入监督区，不能作为Encoder特征。固定验证预览在同目录A_VAL_0/1、B_VAL。

![随机初始化实际预测](preview/dry_run_predictions.png)

每行是一个A/B任务；列为可见输入、真实目标、随机初始化预测、绝对误差及真实评价区域。没有参数更新，图的用途是确认数据流及量纲，不是展示重建学习成功。数组、源输入/预测哈希在cases/index.json。

G3 GPU阶段包含8个评分encode、12个预览encode和2个梯度encode，共22个；原capacity/encoder_views=8仅为评分视图，不包括其余诊断。该口径修正在capacity_scope.json明确，不能将预检视图算作优化更新。CPU补充泄漏检查、图轴顺序修复及固定案例另有源码和记录，原执行源码与原图保留。GPU阶段wall约18.60秒，峰值分配约4.02GiB；A/B初始梯度范数约1987/844，只表示当前初始尺度，不是损失改善。完整成本以dry_run.json与cases/index.json为准。

下一步按用户配置的automation90“完成g3，进入g4并训练”另行进入G4，合计最多200更新、单卡FP32、seed0。G4冻结四TRAIN/四VAL原生观测，微批1累积四源观测，AdamW5e-4、weight decay0.04、clip3，前10步线性warmup后固定学习率；不设置旧的1e-6尾段。采用候选平滑A/B课程，前70步A，后130步加入B并保留A回放；每25步固定评价，另看同训练影像新遮盖，避免把记忆与地域泛化混为一谈。目标组合轮换和实际曝光逐项记账。

G4对A使用同波段四近邻反距离空间插值，对B使用TRAIN-only Ridge/均值；不把空间插值放进整波段B排名。初始、共同检查点、best与last分别保存；第50步磁盘恢复检验非空Adam、RNG、调度、采样位置、曝光和Viewer序号。200步是有界诊断，不代表收敛，不能因此自动进入G5、跨卫星或物理损失训练。

主要复现入口：g3_tasks.py、check_g3_tasks.py、g3_cases.py；验收证据为acceptance.json、viewer_preflight.json及source/final_source_manifest。稳定入口为observation_model/README.md；共享台账未越过其他聊天的认领锁改写。旧训练、旧checkpoint、数据成员与下载队列均保留。
'''
    (G3/'preflight_report.md').write_text(report,encoding='utf-8')
    G4.mkdir(exist_ok=True)
    authorization=dict(authorized=True,stage='G4',permitted_optimizer_updates=200,seed=0,
        source='User-configured heartbeat automation90',instruction='完成g3，进入g4并训练。如果已在训练则等待不做任何操作',
        automation_config_sha256=digest(Path('C:/Users/yzh/.codex/automations/90/automation.toml')),
        G3_acceptance_sha256=digest(G3/'acceptance.json'),formal_pretraining_authorized=False,
        no_G5_or_cross_source_continuation=True,download_queue_retained=True)
    save(G4/'authorization.json',authorization)
    (G4/'README.md').write_text('# G4有界学习诊断\n\n按用户配置的automation90在G3通过后启动；最多200次优化更新，不能自动进入G5。\n\n实际状态见status.json；配置、成员、源码、基线、初始/逐次固定评价、逐更新曝光、best/last及第50步恢复证据分别保存在本目录。四TRAIN/四VAL仅用于有界能力诊断，不代表全部7395份训练池的泛化。\n\n[任务与预检](../G3_task/preflight_report.md) · [Viewer](http://127.0.0.1:8766/)\n',encoding='utf-8')
    p=ROOT/'observation_model/README.md';body=p.read_text(encoding='utf-8')
    i=body.index('> 当前工作：');j=body.index('\n',i)
    body=body[:i]+'> 当前工作：G3零更新验收通过；已按用户automation90追加授权准备G4最多200步有界诊断。G4实际进度以status.json为准；无G5/跨卫星续行。'+body[j:]
    body+='\n<!-- ENCODER_V2_G3_START -->\n## Encoder V2：G3任务验收与G4入口\n\n[G3预检报告](outputs/encoder_v2/G3_task/preflight_report.md)、[实际任务卡](outputs/encoder_v2/G3_task/task_contract.json)、[Viewer验收](outputs/encoder_v2/G3_task/viewer_preflight.json)与[零更新证据](outputs/encoder_v2/G3_task/acceptance.json)已完成。真实多位置遮盖、整波段删除、均衡损失、候选课程、固定验证及主线程心跳/恢复均有实际产物；0更新。\n\n用户配置的automation90随后明确要求进入G4并训练，授权与200步上限见[G4批准记录](outputs/encoder_v2/G4_fit/authorization.json)，实际是否启动及已完成步数见[运行状态](outputs/encoder_v2/G4_fit/status.json)。G4只用四TRAIN/四VAL完整原生窗口作能力诊断，不能混作全训练池实验；不自动进入G5、跨源或物理损失。\n<!-- ENCODER_V2_G3_END -->\n'
    p.write_text(body,encoding='utf-8')
    save(G3/'delivery_hashes.json',{p.name:digest(p) for p in G3.iterdir() if p.is_file() and p.name!='delivery_hashes.json'})
    print('G3_ACCEPTED_G4_EXPLICIT_200_UPDATE_AUTHORIZATION_PREPARED')


if __name__=='__main__':main()
