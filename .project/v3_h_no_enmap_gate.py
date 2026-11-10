"""No-EnMAP membership, original architecture, two-GPU, recovery and VAL gate."""
import ssl,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('CUDA_LAUNCH_BLOCKING','0')
import sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'earth-representation';sys.path.insert(0,str(ROOT/'src'))
import torch
from encoder_v3.hierarchical.train import load_data,build_model,identity,EpochPlan,batch_loss,update,validate,save,resume,write
from encoder_v3.hierarchical.parallel import TwoDeviceBatch
from encoder_v3.hierarchical.verification import compare_snapshots
from encoder_v3.latent_response.checkpoint import snapshot,parameter_sha
from encoder_v3.shared_mae.model import SharedMemory,ConditionalConvolution


def main():
    out=ROOT/'outputs/encoder_v3/hierarchical/no_enmap/gpu_v1';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/config_v3_h_no_enmap_v1.json').read_text());torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.set_device(config['device'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sets,resolver=load_data(config);full,_=load_data();ds=sets['TRAIN'];plan=EpochPlan(ds.rows,17,32)
    assert len(ds.rows)==7395 and len(sets['VAL'].rows)==383 and len(plan)==232 and len(plan.batch(231))==3
    assert {r['id'] for r in ds.rows}=={r['id'] for r in full['TRAIN'].rows if r['sensor']!='EnMAP'}
    assert [r['id'] for r in sets['VAL'].rows]==[r['id'] for r in full['VAL'].rows]
    coverage=[r['id'] for i in range(len(plan)) for r in plan.batch(i)]
    assert len(coverage)==len(set(coverage))==7395 and all(r['sensor']!='EnMAP' for i in range(len(plan)) for r in plan.batch(i))
    model=build_model(config).to(config['device']);initial=parameter_sha(model)
    original_init=json.loads((ROOT/'outputs/encoder_v3/hierarchical/release_a_only_v1/initialization.json').read_text())
    expected=original_init.get('initial_parameter_sha256',original_init.get('parameter_sha256'))
    if expected is None:expected='0958271abbe57a2c505f3716cbdcd1291e253e61d0880d430c43396678b84057'
    assert initial==expected,(initial,expected)
    assert isinstance(model.online.memory,SharedMemory) and isinstance(model.online.convolution,ConditionalConvolution)
    binding=identity(config,sets,resolver);report={'status':'RUNNING','identity':binding,'initial_parameter_sha256':initial,'same_initial_parameters_as_original_A_only':True,
        'TRAIN':7395,'VAL':383,'EnMAP_TRAIN':0,'coverage_unique_once':True,'batches':232,'tail_batch':3,'formal_updates':0,'engineering_updates':0,'pairs':[]}
    worker=TwoDeviceBatch(model,ds,resolver,config,validation_dataset=sets['VAL'])
    assert all(all(r['sensor']!='EnMAP' for r in d.rows) for d in worker.datasets)
    def flush():write(out/'report.json',report)
    def grads():return {n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
    def sync():
        for d in config['devices']:torch.cuda.synchronize(d)
    flush()
    try:
        for cursor in (0,231):
            rows=plan.batch(cursor);model.zero_grad(set_to_none=True);a,ar=batch_loss(model,ds,rows,resolver,config,backward=True);expected_grad=grads()
            sync();start=time.perf_counter();b,br=worker.backward(rows,0);sync();elapsed=time.perf_counter()-start;actual=grads()
            assert expected_grad.keys()==actual.keys() and abs(a-b)<2e-6
            relative=(sum(float((expected_grad[n]-v).square().sum()) for n,v in actual.items())/max(sum(float(v.square().sum()) for v in expected_grad.values()),1e-30))**.5
            assert relative<3e-4 and [r['id'] for r in ar]==[r['id'] for r in br]
            report['pairs'].append({'cursor':cursor+1,'observations':len(rows),'seconds':elapsed,'loss_difference':abs(a-b),'gradient_relative_l2':relative,'sources':sorted({r['sensor'] for r in rows})});flush();print('PAIR_NO_ENMAP',cursor+1,elapsed,flush=True)
        pixels=lambda r:sum(b['shape'][0]*b['shape'][1] for b in r['bands'])
        largest=sorted(ds.rows,key=pixels,reverse=True)[:2];sync();start=time.perf_counter();value,records=worker.backward(largest,0);sync()
        report['capacity']={'seconds':time.perf_counter()-start,'ids':[r['id'] for r in largest],'pixels':[pixels(r) for r in largest],'sources':[r['sensor'] for r in largest],'loss':value,'peak_allocated':{d:torch.cuda.max_memory_allocated(d) for d in config['devices']}}
        assert parameter_sha(model)==initial;flush();print('CAPACITY_NO_ENMAP',report['capacity']['seconds'],flush=True)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay']);scheduler=torch.optim.lr_scheduler.StepLR(opt,100,.5);state={'cursor':0};runtime=dict(config,batches_per_epoch=len(plan))
        update(model,ds,plan.batch(0),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
        save(out/'recovery.pt',model,opt,scheduler,state,binding)
        update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1;expected_state=snapshot(model,opt,state,scheduler)
        resume(out/'recovery.pt',model,opt,scheduler,state,binding)
        update(model,ds,plan.batch(1),resolver,opt,scheduler,state,runtime,parallel=worker);report['engineering_updates']+=1
        report['complete_next_update_restored']=compare_snapshots(expected_state,snapshot(model,opt,state,scheduler))
        before=snapshot(model,opt,state,scheduler);start=time.perf_counter();val=validate(model,sets['VAL'],resolver,opt,scheduler,state,runtime,parallel=worker)
        report['validation']={'observations':val['observations'],'metric':val['metric'],'seconds':time.perf_counter()-start}
        assert val['observations']==383 and all(set(r['tasks'])=={'A'} for r in val['members'])
        report['validation_state_restored']=compare_snapshots(before,snapshot(model,opt,state,scheduler))
        report['status']='PASS'
    except BaseException as exc:report.update(status='FAILED',error=repr(exc));raise
    finally:worker.close();flush()
    print('NO_ENMAP_GPU_PASS',flush=True)


if __name__=='__main__':main()
