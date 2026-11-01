"""Regenerate the stable research report, curves and fixed GT-selected cases."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/verification"
plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
NAMES={"random":"随机共享单波段CNN", "baseline_verified":"共享单波段CNN｜标准增强", "ordinary_verified":"共享单波段CNN｜普通缩放", "physical_verified":"共享单波段CNN｜低通后采样", "dofa_external":"DOFA官方预训练ViT-B", "DOFA":"DOFA官方预训练ViT-B"}
for mode,title in [("uniform","全区域"),("random","随机区域"),("selective","修正选择性")]:NAMES["corrective_scale_"+mode]="共享单波段CNN｜"+title+"尺度约束"
for mode,title in [("single","S2单源双组视图"),("mixed","两源独立多视图"),("paired","两源配对正视图")]:NAMES["sensor_"+mode]="共享单波段CNN｜"+title
for tag,title in [("legacy_off3k","历史标准DINO｜3000步"),("legacy_mixed2k","历史三源混合｜2000步"),("legacy_e1_0","历史标准增强｜2000步"),("legacy_e1_1","历史普通缩放｜2000步"),("legacy_e1_2","历史低通后采样｜2000步"),("legacy_e2_0","历史无尺度约束｜1500步"),("legacy_e2_3","历史随机尺度约束｜1500步"),("legacy_e2_4","历史选择性尺度约束｜1500步")]:NAMES[tag]=title
MAP={"verified_s2_standard":"baseline_verified","verified_s2_generic":"ordinary_verified","verified_s2_physical":"physical_verified"}
def load(name):return json.loads((OUT/name).read_text(encoding="utf-8"))
def name(tag):return NAMES.get(MAP.get(tag,tag),tag)
def f(value):return "—" if value is None else f"{value:.4f}"
def table(headers,rows):return "| "+" | ".join(headers)+" |\n|"+"|".join(["---"]*len(headers))+"|\n"+"\n".join("| "+" | ".join(map(str,row))+" |" for row in rows)+"\n"


def architecture():
    fig,ax=plt.subplots(figsize=(13,5.5));ax.set(xlim=(0,13),ylim=(0,5.5));ax.axis("off")
    nodes={"data":(1.5,4.4,"浮点反射率 + 有效像元\n波长、原生GSD、可用波段"),"stem":(5,4.4,"共享单波段CNN + FiLM\n只编码有效波段"),"enc":(10,4.4,"波段集合均值 → ResNet-18\n四级空间特征 / 全局特征"),"dino":(10,2.4,"学生 / EMA教师 + DINO头\n同源或配对多视图目标"),"scale":(5,2.4,"可选尺度约束：共同320m支持\n输入退化权重，固定梯度"),"head":(10,.6,"冻结编码器 → 同一轻量任务头\n随后基线与最佳新候选微调")}
    for x,y,text in nodes.values():ax.text(x,y,text,ha="center",va="center",bbox=dict(boxstyle="round,pad=.6",fc="#edf3f8",ec="#45708c"),fontsize=11)
    def arrow(a,b,start=(0,0),end=(0,0)):
        x,y,_=nodes[a];xx,yy,_=nodes[b];ax.annotate("",xy=(xx+end[0],yy+end[1]),xytext=(x+start[0],y+start[1]),arrowprops=dict(arrowstyle="->",color="#345568",lw=1.7))
    arrow("data","stem",(1.6,0),(-1.5,0));arrow("stem","enc",(1.7,0),(-2.2,0));arrow("enc","dino",(0,-.5),(0,.5));arrow("enc","head",(2,-.4),(2,.35));arrow("enc","scale",(-1.6,-.5),(1.9,.4));arrow("data","scale",(0,-.55),(-1.6,.4))
    fig.tight_layout();fig.savefig(OUT/"model_connections.png",dpi=160);plt.close(fig)


def curves(training):
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for group,tags in enumerate([[r["tag"] for r in training if not r["tag"].startswith("sensor_")],[r["tag"] for r in training if r["tag"].startswith("sensor_")]]):
        for tag in tags:
            rows=[json.loads(s) for s in (OUT/"tvlp"/tag/"metrics.jsonl").read_text().splitlines()]
            # Resume can repeat the last few updates; keep the last record per step.
            tr={r["global_step"]:r for r in rows if r["phase"]=="train"};vr={r["global_step"]:r for r in rows if r["phase"]=="val"}
            for col,key,source in [(0,"loss",vr),(1,"lr",tr),(2,"exposure",tr)]:
                steps=sorted(source);axes[group,col].plot(steps,[source[s]["metrics"][key] for s in steps],label=name(tag).replace("共享单波段CNN｜",""),lw=1.1)
        for col,title in enumerate(["固定S2视图：无标签验证损失","学习率","累计位置曝光"]):
            ax=axes[group,col];ax.set(title=("980位置队列｜" if group==0 else "290位置队列｜")+title,xlabel="实际更新次数");ax.grid(alpha=.2)
            if col==1:ax.set_yscale("log")
        axes[group,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"training_curves.png",dpi=150);plt.close(fig)


def cases(probes,candidate):
    rows=[json.loads(s) for s in (ROOT.parent/"landslide-seg/manifests/v4r1/pool_100.jsonl").read_text().splitlines()]
    rows=[r for r in rows if r["split"]=="val"]
    selected=[]
    for event in ["hiroshima","thrissur"]:
        options=[]
        for row in rows:
            if row["event_id"]!=event:continue
            with np.load(row["cache"]) as z:
                mask=(z["mask"].T==1)&z["valid"].T.astype(bool)
                components,n=label(mask,structure=np.ones((3,3)))
                largest=np.bincount(components.ravel())[1:].max() if n else 0
            if largest:options.append((largest,row))
        for target in [20,150]:
            _,row=min(options,key=lambda pair:(abs(np.log(pair[0]/target)),pair[1]["sample_id"]))
            selected.append(row);options=[p for p in options if p[1]["sample_id"]!=row["sample_id"]]
    (OUT/"fixed_case_ids.json").write_text(json.dumps(dict(selection="per event largest GT component closest to20 or150pixels in log space; independent of predictions",ids=[r["sample_id"] for r in selected]),indent=2))
    tags=["random_frozen","baseline_verified",candidate,"physical_verified","corrective_scale_selective","sensor_paired","dofa_external"]
    tags=list(dict.fromkeys(tags));preds={}
    for tag in tags:
        with np.load(OUT/"probes"/tag/"predictions.npz") as z:preds[tag]={str(s):p for s,p in zip(z["ids"],z["best"])}
    fine=load("finetune/"+candidate+"/result.json")
    with np.load(OUT/"finetune"/candidate/"predictions.npz") as z:preds["finetuned"]={str(s):p for s,p in zip(z["ids"],z["best"])}
    tags.append("finetuned")
    for binary in [False,True]:
        fig,axes=plt.subplots(len(selected),2+len(tags),figsize=(2*(2+len(tags)),2.5*len(selected)))
        for i,row in enumerate(selected):
            sid=row["sample_id"]
            with np.load(row["cache"]) as z:
                valid=z["valid"].T.astype(bool);gt=(z["mask"].T==1)&valid;rgb=np.clip(z["x10"][[2,1,0]].transpose(2,1,0)/3000,0,1)
            axes[i,0].imshow(rgb);axes[i,0].set_ylabel(sid,fontsize=8)
            gray=plt.get_cmap("gray").copy();gray.set_bad("#888888")
            heat=plt.get_cmap("magma").copy();heat.set_bad("#888888")
            axes[i,1].imshow(np.ma.masked_where(~valid,gt),vmin=0,vmax=1,cmap=gray)
            for j,tag in enumerate(tags,2):
                p=preds[tag][sid];axes[i,j].imshow(np.ma.masked_where(~valid,p>=.5 if binary else p),vmin=0,vmax=1,cmap=gray if binary else heat)
                if gt.any():axes[i,j].contour(gt,levels=[.5],colors="cyan",linewidths=.5)
            for ax in axes[i]:ax.set(xticks=[],yticks=[])
        titles=["S2 RGB（固定亮度）","人工滑坡标签"]+["最佳新候选微调" if t=="finetuned" else name("random" if t=="random_frozen" else t).replace("共享单波段CNN｜","") for t in tags]
        for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=9)
        fig.suptitle("阈值0.5二值预测" if binary else "滑坡概率：黑=0，亮黄=1；青线=人工标签边界",fontsize=12)
        fig.tight_layout();fig.savefig(OUT/("fixed_cases_binary.png" if binary else "fixed_cases_probability.png"),dpi=145);plt.close(fig)


def main():
    training=load("training_acceptance.json")["runs"]
    probes=load("probe_comparison.json");p={r["tag"]:r for r in probes}
    current=[r for r in probes if not r["tag"].startswith("legacy_") and r["tag"] not in ["random_frozen","dofa_external"]]
    candidate=max(current,key=lambda r:r["best_event_mean_iou"])["tag"]
    geographic={r["tag"]:r for r in load("geographic_retrieval.json")["results"]}
    weak=load("worldcover_comparison.json")
    architecture();curves(training);cases(probes,candidate)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    route_tags=["sensor_single","sensor_mixed","sensor_paired"]
    for ax,values,title in [(axes[0],[geographic[t]["S2_to_Landsat"]["recall_at_10"] for t in route_tags],"地域开发：跨卫星R@10"),(axes[1],[p[t]["best_event_mean_iou"] for t in route_tags],"人工滑坡标签：冻结头IoU")]:
        ax.bar(["S2单源","两源独立多视图","配对正视图"],values,color=["#7496ac","#64a398","#c78c65"]);ax.set(title=title,ylim=(0,max(values)*1.25));ax.grid(axis="y",alpha=.2)
        for i,v in enumerate(values):ax.text(i,v,f(v),ha="center",va="bottom")
    fig.tight_layout();fig.savefig(OUT/"sensor_task_comparison.png",dpi=150);plt.close(fig)
    mainrows=[]
    for r in training:
        tag=r["tag"];pr=p[MAP.get(tag,tag)];s=r["checkpoints"][0];o=pr["overall"]
        mainrows.append([name(tag),f'{s["epoch"]} / {s["step"]}',str(s["exposure"]),f(r["common_step_fixed_view_dino_loss"]),f(pr["best_event_mean_iou"]),f(pr["last_event_mean_iou"]),f(o["boundary"]["f1"]),f(o["objects"]["bins"]["small"]["object_recall_evaluable"]),f(geographic[tag]["S2_to_Landsat"]["recall_at_10"])])
    for pt,gt in [("random_frozen","random"),("dofa_external","DOFA")]:
        r=p[pt];o=r["overall"];mainrows.append([name(gt),"外部参照","—","—",f(r["best_event_mean_iou"]),f(r["last_event_mean_iou"]),f(o["boundary"]["f1"]),f(o["objects"]["bins"]["small"]["object_recall_evaluable"]),f(geographic[gt]["S2_to_Landsat"]["recall_at_10"])])
    weakrows=[[name(r["tag"]),f(r["classifier"]["test_map"]),f(r["segmentation"]["test_miou"])] for r in weak if not r["tag"].startswith("legacy_")]
    histrows=[]
    for r in probes:
        if r["tag"].startswith("legacy_"):histrows.append([name(r["tag"]),f(r["best_event_mean_iou"]),f(r["overall"]["boundary"]["f1"]),f(next((x["classifier"]["test_map"] for x in weak if x["tag"]==r["tag"]),None)),f(next((x["segmentation"]["test_miou"] for x in weak if x["tag"]==r["tag"]),None))])
    hls=load("hls_feature_reference.json")["results"]
    hlsrows=[[{"spectral_statistics":"反射率统计","verified_CNN":"修复标准CNN","DOFA":"DOFA官方预训练"}[r["model"]],f(r["raw"]["recall@1"]),f(r["harmonized"]["recall@1"]),f(r["raw"]["recall@10"]),f(r["harmonized"]["recall@10"])] for r in hls]
    ft0=load("finetune/baseline_verified/result.json");ft=load("finetune/"+candidate+"/result.json")
    text="""# 跨卫星表征：修复、真实评价与原计划接续

## 阅读约定与实验设置

本报告区分修复后新实验、历史权重和外部预训练参照。共享单波段CNN逐波段编码，以中心波长和原生地面分辨率调制特征；只编码实际可用且未被随机删除的波段，集合均值接入ResNet-18。学生预测多视图，教师由EMA更新，不使用滑坡或土地覆盖标签做自监督预训练。

自监督种子0，位置batch32，FP32，AdamW初始学习率5e-4、正则化参数weight decay0.04、梯度裁剪3。前10轮warmup，此后固定S2开发视图的无标签DINO损失连续10轮相对改善不足0.5%时减半，最低1e-6；至少100轮，30轮无改善且学习率≤初始1/16时早停，上限400轮或50000次更新。最佳checkpoint采用相对改善门槛，**不是每次记录的绝对最低损失**；另存末态。不同分支独立早停，最佳任务分数不是严格等预算消融。没有运行BF16，因而无需追加精度矩阵。

980位置队列：从九个训练事件2223个候选按实际所选时相质量筛得980块、7个事件；无标签开发验证36块来自Hiroshima。290位置队列：在980块中再按两源共同QA≥80%、日期差≤2天、共同30m NDVI相关≥0.6、差异≤0.2、零位移相对±1像元最优相关不落后超过0.05筛选，保留Chimanimani220、Indonesia29、Kyrgyzstan1 34、Kyrgyzstan2 7。后者是稳定且有纹理的子集，不代表所有地表。

S2单源双组视图、两源独立多视图、两源配对正视图均用同一290位置、8学生视图和4教师视图。第一组复制同一S2观测，不计为两次独立观测；后两组每位置包含S2和Landsat，只有DINO目标配对集合由12项改为28项。各组每批32位置，后两组实际有64个传感器观测。新companion共954个Landsat、1014个HLS，均存新版目录，旧数据未覆盖。

滑坡任务使用**我们的数据划分（留事件验证）**：九个训练事件2223块，开发评价Hiroshima448+Thrissur221=669块，保持原清单与stats_core25文件。这里输入物理反射率，不套用core25标准化；冻结特征只用训练集拟合逐通道均值方差。轻量解码头、阈值0.5、种子1234相同；按两个开发事件等权IoU选头。开发集同时用于选轮次与候选，不能叫独立测试。下表IoU为两事件等权；边界F1与小对象召回为669块合并计数，边界容差1个10m像元，小对象<50像元且符合项目可评价对象定义。原生10m标签诊断与共同30m产品比较分开。

跨卫星地理开发检索：18个Hiroshima查询，图库290训练位置+18开发位置=308。为获得足够晴空配对，从36个原开发地点的现有15时相中仅按SCL清晰度选时相，再按上述配对门槛筛到18组；实际日期写入独立清单，没有替换训练数据。精确并列相似度取随机打破并列时的期望命中率。R@10是正确位置进入前10名的比例，随机排名期望10/308=0.0325；邻近640m召回单独存JSON。该开发诊断只含一个事件，不能外推全球。

## 主表：新训练、真实滑坡标签与跨卫星检索

共同步数列仅比较固定无标签视图损失：980队列取3100步，290队列取1210步，两队列分开解读；没有这些步数的下游checkpoint，不能称为共同步数任务精度。曝光表示位置参与更新的次数，包含最后不足32的batch。

"""
    text+=table(["模型与机制","轮数 / 更新","位置曝光","共同步数DINO损失","最佳头IoU","末态头IoU","边界F1","小对象召回","地理开发R@10"],mainrows)
    text+="\n表中位置曝光按最终保留的优化轨迹计数。配对组中断前有10次更新（290次位置曝光）未进入最近checkpoint，恢复后重做；实际尝试2650次更新、76850次位置曝光，最终保留2640次更新、76560次位置曝光。成本统计包含丢弃的尝试。\n"
    text+="\n## 图与解释\n\n![组件连接](../outputs/verification/model_connections.png)\n\n从反射率和有效波段进入共享编码器；无标签DINO头、可选尺度目标、下游任务头分别消费编码器特征。尺度权重来自输入退化残差并固定梯度，比较高分辨率s16特征与低通后20m输入的s16特征，统一到320m地面支持；随机区域与选择性区域匹配数量和权重总量，额外前向不更新BN运行统计。随机区域分支额外消耗增强随机数，因此不宣称逐裁剪完全相同。\n\n![曲线](../outputs/verification/training_curves.png)\n\n上排是980位置队列，下排是290位置队列；三列分别为固定开发视图损失、学习率和位置曝光，横轴实际更新次数。低DINO损失不是语义精度，独立早停也不提供等预算因果结论。\n\n![固定案例概率](../outputs/verification/fixed_cases_probability.png)\n\n![固定案例二值](../outputs/verification/fixed_cases_binary.png)\n\n每行按人工标签最大连通对象接近20或150像元选取，每个开发事件两块，与模型预测无关；所有列使用相同地点。前两列是RGB和人工标签，其后为不同模型概率或阈值0.5预测，青线为人工边界。应观察误报、细小对象和边缘覆盖，不能用四个案例替代全量指标。\n\n## 本地WorldCover补充诊断\n\n现有中国森林地区100块S2影像按25个地域分组，60训练/20验证/20测试（15/5/5地域）；排除已知预训练区域邻域后无样本被剔除。使用原始WorldCover地图，不用NDVI森林掩码和合成火灾标签。影像与评价网格20m，原地图10m产品被重采样，不证明原生10m细节能力。土地覆盖分类是多标签：某类别占有效区≥1%即为正例，只保留训练中至少3正3负的草地、农田、建成区、苔藓地衣；测试苔藓地衣无正例，mAP只平均前三类，测试正例17/2/2。小样本与地图模型误差都会影响结果。分类和分割来自同一地图源，不是两个独立真值数据集。\n\n"
    text+=table(["模型与机制","地图多标签分类mAP","地图分割mIoU"],weakrows)
    text+="\n## 历史权重复评：按原预算单列\n\n历史编码器统一冻结在本轮输入和相同任务头上评价。历史DINO循环与数据处理存在已修复错误，原标称更新步数不等于正确曝光；部分区域可能已参与历史无标签预训练，无法完整确认的接触范围保留限制。它们是历史诊断，不与新队列混作等预算消融。\n\n"
    text+=table(["历史结构与标称预算","滑坡IoU","边界F1","地图分类mAP","地图分割mIoU"],histrows)
    text+="\n历史九条件的136位置区域检索完整保留在[原结果JSON](../outputs/verification/existing_retrieval.json)，区域是Chimanimani且预训练可能接触，不是地理留出。原区域随机编码器S2→Landsat/HLS R@10均0.3824，修复标准编码器相应0.2206/0.3382；随机编码器的滑坡任务表现却极低。区域检索可被低层输入统计支持，不能代替语义任务。不同图库与并列处理的数值不混排。\n\n## 同源产品控制与HLS接续\n\n![同源产品控制](../outputs/verification/same_acquisition_control.png)\n\n每行一个共同30m窗口，列为S2 L2A、同次采集HLS S30 NDVI及差异。S2同源相关中位数0.9832，平移1像元0.7863、错地点0.0060；Landsat/L30相应0.9901、0.8248、0.0071。两组各20窗口、各一次独立采集，同一景多个窗口不当作独立场景。近时跨卫星另40对相关中位数0.9130，错地点0.0290。它们证明修复链能区分配对，不能证明表征学到了通用语义。\n\n![近时配对控制](../outputs/verification/pair_controls.png)\n\n横轴是同地点、错位、转置和错地点，纵轴共同30m NDVI相关；应看正负控制能否分开。源产品、实际日期和共同网格见控制JSON。HLS来源与原产品元数据已关联，S30与L30在相同20个30m窗口、共同六波段上比较如下。原始组是S2 L2A→Landsat L2，协调组是S30→L30；四产品来自两个采集日期，不能视作四个独立观测。\n\n"
    text+=table(["特征","原始产品R@1","HLS协调R@1","原始R@10","协调R@10"],hlsrows)
    text+="\n## 展开结论\n\n"
    baseline=p["baseline_verified"];ordinary=p["ordinary_verified"];physical=p["physical_verified"]
    text+=f"共享编码器在滑坡监督头下具有可利用的空间信息：随机编码器IoU {f(p['random_frozen']['best_event_mean_iou'])}，修复标准编码器{f(baseline['best_event_mean_iou'])}。但随机编码器的地域检索高于多数训练编码器，说明检索机制与任务语义存在差别；具体由何种低层统计主导仍未验证。该结论由主表和相同固定案例共同支持。\n\n"
    text+="![跨卫星目标与下游任务](../outputs/verification/sensor_task_comparison.png)\n\n左图是相同18查询、308图库的跨卫星检索，右图是相同669块人工滑坡标签的冻结头IoU；横轴对应同一290位置训练的三个目标。配对正视图把S2→Landsat R@1/R@10提高到0.6111/0.7778，两源独立多视图均为0，但滑坡IoU相应0.0744/0.1044，未同时改善。配对目标确实在此次地域开发检索中产生作用，不能据此推断通用地表语义更强。各组独立早停、配对子集偏向稳定纹理且只有一个开发事件，不宣称统计显著性。\n\n"
    text+=f"物理退化没有在本次真实辅助任务中显示优势：强度匹配后，普通缩放IoU {f(ordinary['best_event_mean_iou'])}、低通后采样{f(physical['best_event_mean_iou'])}，标准{f(baseline['best_event_mean_iou'])}。尺度分支也未显示一致的IoU、边界和小对象收益；因此停止追加波段子采样或辐射扰动，不继续堆叠损失。曲线与固定案例展示了损失改善不能直接等同下游收益。单种子与独立早停限制了因果解释。\n\n"
    text+=f"按相同669块开发标签，最佳可执行新CNN候选为{name(candidate)}。同一微调策略下，基线最佳/末态IoU {f(ft0['best_event_mean_iou'])}/{f(ft0['last_event_mean_iou'])}，候选{f(ft['best_event_mean_iou'])}/{f(ft['last_event_mean_iou'])}，最佳/末态权重分别保存。候选按本开发集选择，此处提升不是独立测试证据。DOFA使用本地官方111.3M参数编码器，相比本CNN约11.7M（含DINO训练头），预训练规模、数据和结构均不同；相同任务头与评价集合提供外部参照，不能归因于单一组件。\n\n"
    text+="## 限制与下一步\n\n本轮已执行具备条件的S2/两源训练、一次物理增强比较、一次修正尺度比较、HLS同源关联和共同30m评价、本地DOFA、冻结任务头及基线/新候选微调。35项已有与关键路径测试通过；100步GPU工程试跑在第50步恢复，另有混合传感器多视图和尺度梯度小批验证。没有扩展防御框架或多种子队列。历史数据与checkpoint保留，修复版本单独存放。\n\n两源独立多视图在Windows CUDA初始化时两次原生失败，0次更新；提前初始化CUDA后完成。配对组第2220步附近停止心跳且GPU上下文消失，确认后只终止本任务工作进程，从2210步checkpoint恢复完成。原始失败与恢复记录均保留，原因未确认，不伪称全程无故障。续训模型、优化器、调度器、随机状态、采样位置和早停状态已保存；早期三个980位置运行的轮次边界RNG另存修正副本，原权重不变。\n\n**仍缺独立人工土地覆盖分类与分割真值**，对应原计划结论保持受阻，不自动下载大型数据。WorldCover只是补充产品一致性。滑坡标签可支撑10m辅助边界与小对象评价，不能替代原土地覆盖目标。HLS20窗口是工程参考，不能据此确认大范围协调收益。DOFA预训练地理接触未知。EnMAP保持可选，不重试已有认证失败。\n\n接续顺序：取得独立土地覆盖标签后，在冻结的地域划分上先比较标准CNN、两源三组、DOFA的同任务头，再仅微调基线与冻结阶段选中的候选；共同30m任务与真实10m细节分开。当前物理增强缺乏真实任务收益，补充增强分支停止；尺度只完成本轮修正，不继续扫描超参数。此为条件计划，不假装已执行缺失任务。\n\n## 附录：配置、成本与产物索引\n\n"
    text+="- [实际配置、最佳/末态、曝光、共同步数损失、关键权重哈希和日志耗时](../outputs/verification/training_acceptance.json)。日志活跃时长为各会话首末训练记录差之和，排除初始化和中断等待，不能叫纯GPU计算时间；早期冻结版本未记录SSL峰值显存，明确缺项，未倒填估计数。辅助头与微调result.json记录实测耗时、峰值分配显存。新入口已补充实际显存和会话吞吐日志。\n- [固定执行配置](../configs/verified_workflow.json)，[DINO固定版本](../configs/dino_source.json)，[依赖环境](../outputs/verification/environment.json)。各运行provenance保留代码哈希、实际参数和成员；冻结源码分别位于release、augmentation_release、scale_release、sensor_code_release。\n- [滑坡逐事件、头初始化与成本](../outputs/verification/probe_comparison.json)，[地理检索全部方向和逐查询](../outputs/verification/geographic_retrieval.json)，[WorldCover结果](../outputs/verification/worldcover_comparison.json)，[微调目录](../outputs/verification/finetune)。\n- [原始产品来源与S30控制](../outputs/verification/same_acquisition_control.json)，[L30控制](../outputs/verification/hls_l30_control.json)，[同网格六波段HLS比较](../outputs/verification/hls_feature_input_manifest.json)，[新版companion数值审计](../outputs/verification/companion_audit.json)。\n- [模型导出清单](../outputs/verification/export/manifest.json)，[运行与条件状态](../outputs/verification/workflow_state.json)，[Viewer校验](../outputs/verification/viewer_validation.json)。\n- Training Viewer：[打开本机查看器](http://127.0.0.1:8766/)。TVLP 1.0日志包含主线程心跳、恢复会话、真实损失/学习率/曝光；固定案例以artifact登记。旧版本缺失的资源字段保留缺项，完成后不伪造训练时采样值。复现命令见项目README。\n"
    cost=f"九个自监督运行的日志活跃时长合计{sum(r['active_logged_seconds'] for r in training)/60:.1f}分钟，最终保留{sum(r['checkpoints'][0]['step'] for r in training)}次更新、{sum(r['checkpoints'][0]['exposure'] for r in training)}次位置曝光；含回滚的实际尝试曝光见验收JSON。并发运行且GPU有共享使用，这不是独占GPU核时。所有新CNN从种子0随机初始化，没有加载历史权重；轻量任务头种子1234随机初始化。DOFA加载本地官方预训练权重；四级投影各32通道与后续融合头规则相同，投影层因编码器通道不同而参数量不同。\n\n"
    text=text.replace("## 附录：配置、成本与产物索引\n\n","## 附录：配置、成本与产物索引\n\n"+cost)
    text=text.replace("青线为人工边界。","青线为人工边界，灰色为无效像元。")
    text=text.replace("此后固定S2开发视图的无标签DINO损失","此后固定S2开发视图的无标签DINO损失（实际为32与4样本两批等权平均）")
    (ROOT/"reports/verification.md").write_text(text,encoding="utf-8")
    print(json.dumps(dict(report="reports/verification.md",candidate=candidate),ensure_ascii=False))


if __name__=="__main__":main()
