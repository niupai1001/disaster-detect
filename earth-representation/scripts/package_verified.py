"""Export inference encoders and attach real, completed-run comparison artifacts."""
import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

import torch

from scripts.accept_verified_runs import ROOT,OUT


def main():
    summary=json.loads((OUT/"training_acceptance.json").read_text())
    directory=OUT/"export";directory.mkdir(exist_ok=True)
    exports=[]
    for row in summary["runs"]:
        tag=row["tag"];chosen=row["checkpoints"][1];source=ROOT/chosen["path"]
        path=directory/(tag+".encoder.pt")
        if not path.exists():
            state=torch.load(source,map_location="cpu",weights_only=False)
            weights={k.removeprefix("backbone.encoder."):v for k,v in state["teacher"].items() if k.startswith("backbone.encoder.")}
            torch.save(dict(encoder=weights,source_checkpoint_sha256=chosen["sha256"],architecture=dict(width=32,aggregate="mean"),input="float32 reflectance; centre wavelength nm, native GSD m, boolean available band mask"),path)
            del state,weights
        exports.append(dict(tag=tag,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_checkpoint=chosen["path"],source_sha256=chosen["sha256"],kind="self_supervised_EMA_encoder_no_DINO_head"))
        run=OUT/"tvlp"/tag
        events=[json.loads(s) for s in (run/"events.jsonl").read_text().splitlines()]
        existing={e.get("path") for e in events if e["type"]=="artifact"}
        artifacts=run/"artifacts";artifacts.mkdir(exist_ok=True)
        with (run/"events.jsonl").open("a",encoding="utf-8") as stream:
            for filename in ["fixed_cases_probability.png","fixed_cases_binary.png","training_curves.png","sensor_task_comparison.png"]:
                relative="artifacts/"+filename
                shutil.copyfile(OUT/filename,artifacts/filename)
                if relative in existing:continue
                stream.write(json.dumps(dict(type="artifact",timestamp=datetime.now(timezone.utc).isoformat(),session_id=events[-1]["session_id"],path=relative,artifact_type="image",tag="completed-model-comparison",phase="val",role="comparison",group_id="fixed_GT_selected_cases",metadata=dict(generated_after_training=True,selection_file="outputs/verification/fixed_case_ids.json")),ensure_ascii=False)+"\n")
        meta=json.loads((run/"run.json").read_text())
        meta["metric_definitions"].update({"summary/active_logged_locations_per_second":{"direction":"none"},"summary/final_exposure":{"direction":"none"}})
        (run/"run.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
        metrics=[json.loads(s) for s in (run/"metrics.jsonl").read_text().splitlines()]
        if not any(m["phase"]=="summary" for m in metrics):
            with (run/"metrics.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(dict(session_id=events[-1]["session_id"],timestamp=datetime.now(timezone.utc).isoformat(),phase="summary",global_step=row["checkpoints"][0]["step"],metrics=dict(active_logged_locations_per_second=row["locations_per_second"],final_exposure=row["checkpoints"][0]["exposure"])))+"\n")
    fine=[]
    for p in sorted((OUT/"finetune").glob("*/result.json")):
        result=json.loads(p.read_text());fine.append(dict(tag=result["tag"],kind="supervised_auxiliary_finetune_separate_from_SSL",best=str((p.parent/"best.pt").relative_to(ROOT)),last=str((p.parent/"last.pt").relative_to(ROOT))))
    (directory/"manifest.json").write_text(json.dumps(dict(encoders=exports,supervised_finetuning=fine,boundary="No claim of independent landcover validation. SSL and supervised fine-tuned weights are separate."),indent=2))
    (directory/"README.md").write_text('''# 已验证编码器导出

这些文件只包含无标签EMA教师的UnifiedEncoder权重，移除了DINO头；有监督微调权重单独保留在finetune目录。

```python
import torch
from src.models.unified_encoder import UnifiedEncoder
model = UnifiedEncoder(width=32, aggregate="mean").cuda().eval()
bundle = torch.load("outputs/verification/export/verified_s2_standard.encoder.pt", weights_only=False)
model.load_state_dict(bundle["encoder"])
# image为物理反射率[B,C,H,W]，centre为nm，gsd为原生米数，band_mask为[B,C]可用布尔值。
with torch.no_grad():
    features = model(image.cuda(), centre.cuda(), gsd.cuda(), band_mask.cuda())
```

源checkpoint、SHA256和实际预算见manifest.json与training_acceptance.json。反射率不要重复缩放，填充值必须先进入有效像元mask。10m网格上的Landsat并没有10m原生信息。
''',encoding="utf-8")
    print(json.dumps(dict(exported=len(exports),finetuned=len(fine))))


if __name__=="__main__":main()
