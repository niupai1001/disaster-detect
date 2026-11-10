"""Read-only native module timing evidence, with CUDA events and cold/warm runs."""
import ssl
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys
from pathlib import Path
import json
import time
from collections import defaultdict
import torch

root=Path(__file__).resolve().parents[1]/'earth-representation'
sys.path.insert(0,str(root/'src'))
from encoder_v3.hierarchical.train import load_data,sample_from_dataset,build_model,identity,write
from encoder_v3.hierarchical.model import SparseAttention,GlobalBlock

config=json.loads((root/'configs/config_v3_h_mae_v1.json').read_text())
output=root/'outputs/encoder_v3/hierarchical/engineering/module_profile_v1.json'
if output.exists():raise ValueError('Evidence already exists')
torch.set_num_threads(2);torch.manual_seed(config['seed'])
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
datasets,resolver=load_data();dataset=datasets['TRAIN'];model=build_model(config).to(config['device'])
binding=identity(config,datasets,resolver);results=[]
for sensor in ('S2','EnMAP'):
    candidates=[(i,r) for i,r in enumerate(dataset.rows) if r['sensor']==sensor]
    i,row=sorted(candidates,key=lambda pair:sum(b['shape'][0]*b['shape'][1] for b in pair[1]['bands']))[len(candidates)//2]
    start=time.perf_counter();sample=sample_from_dataset(dataset,i,resolver,config['device']);torch.cuda.synchronize()
    loading=time.perf_counter()-start
    start=time.perf_counter();layout=model.layout(sample);torch.cuda.synchronize();geometry=time.perf_counter()-start
    for repeat in range(3):
        events=defaultdict(list);starts=defaultdict(list);handles=[]
        def before(name):
            def hook(*args):
                event=torch.cuda.Event(enable_timing=True);event.record();starts[name].append(event)
            return hook
        def after(name):
            def hook(*args):
                end=torch.cuda.Event(enable_timing=True);end.record();events[name].append((starts[name].pop(),end))
            return hook
        for name,module in model.named_modules():
            if isinstance(module,(SparseAttention,GlobalBlock)) or name in ('online.memory','online.convolution','output'):
                handles += [module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name),always_call=True)]
        originals=[]
        def instrument(obj,attribute,name):
            original=getattr(obj,attribute)
            def wrapped(*args,**kwargs):
                before(name)()
                try:return original(*args,**kwargs)
                finally:after(name)()
            originals.append((obj,attribute,original));setattr(obj,attribute,wrapped)
        instrument(model.online,'native','native_frontend_inclusive')
        instrument(model.online,'fuse','channel_fusion')
        seen=set()
        for maps in layout[2]:
            for level,mapping in enumerate(maps):
                if id(mapping) not in seen:
                    instrument(mapping,'aggregate',('fine','middle','coarse')[level]+'_area_aggregation');seen.add(id(mapping))
        torch.cuda.reset_peak_memory_stats();model.zero_grad(set_to_none=True)
        first=torch.cuda.Event(enable_timing=True);middle=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();first.record();loss,details=model([sample],kind='A',seed=config['seed']);middle.record();loss.mean().backward();last.record();torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        for h in handles:h.remove()
        for obj,attribute,original in originals:setattr(obj,attribute,original)
        results.append(dict(sensor=sensor,id=row['id'],repeat=repeat,cache='cold' if repeat==0 else 'warm',load_seconds=loading,
            geometry_seconds=geometry,wall_seconds=elapsed,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),
            forward_cuda_ms=first.elapsed_time(middle),backward_cuda_ms=middle.elapsed_time(last),
            inclusive_cuda_ms={k:sum(a.elapsed_time(b) for a,b in v) for k,v in events.items()},
            call_counts={k:len(v) for k,v in events.items()},tokens=details[0]['tokens'],attention_edges=details[0]['attention_edges']))
        write(output,dict(identity=binding,updates=0,results=results,notes='Inclusive module CUDA times include checkpoint recomputation; overlapping parent timings must not be summed. Geometry and data load wall-clock recorded separately. Full Epoch not measured.'))
        print(sensor,repeat,elapsed,flush=True)
