"""Fresh, isolated fixed-TRAIN A learning check with physical RGB errors."""
import ssl,os,sys,json,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('CUDA_LAUNCH_BLOCKING','0')
ROOT=Path(__file__).resolve().parents[1]/'earth-representation';sys.path.insert(0,str(ROOT/'src'))
import torch
from encoder_v3.hierarchical.train import load_data,build_model,identity,sample_from_dataset,update,write
from encoder_v3.latent_response.checkpoint import parameter_sha

def main():
    out=ROOT/'outputs/encoder_v3/hierarchical/a_only/color_learning_probe_v1';out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'configs/config_v3_h_a_only_v1.json').read_text());torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.cuda.set_device(config['device'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sets,resolver=load_data();ds=sets['TRAIN'];pixels=lambda r:sum(b['shape'][0]*b['shape'][1] for b in r['bands'])
    rows=[sorted([r for r in ds.rows if r['sensor']==sensor],key=pixels)[len([r for r in ds.rows if r['sensor']==sensor])//2] for sensor in sorted({r['sensor'] for r in ds.rows})]
    lookup={r['id']:i for i,r in enumerate(ds.rows)};samples=[sample_from_dataset(ds,lookup[r['id']],resolver,config['device']) for r in rows]
    model=build_model(config).to(config['device']);initial=parameter_sha(model)
    def measure():
        result=[]
        with torch.no_grad():
            for row,sample in zip(rows,samples):
                loss,details=model([sample],kind='A',seed=config['seed'],epoch=0,return_predictions=True);detail=details[0]
                ids=[min(range(len(sample.channels)),key=lambda i:abs(sample.channels[i].physical.center_um-wave)) for wave in (.665,.56,.49)]
                bank=ds.datasets[row['sensor']=='EnMAP'];bands=[]
                for i in ids:
                    c=sample.channels[i];mask=detail['target_masks'][i];stat=bank.statistics['bands'][row['instrument']+':'+c.name]
                    error=(detail['predictions'][i]-c.values)[mask]*stat['std']
                    bands.append({'band':c.name,'physical_mae':float(error.abs().mean()),'physical_bias':float(error.mean())})
                result.append({'id':row['id'],'source':row['sensor'],'A_mse':float(loss[0]),'RGB':bands})
        return result
    before=measure();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=config['weight_decay']);scheduler=torch.optim.lr_scheduler.StepLR(opt,1000,.5);state={'cursor':0};runtime=dict(config,batches_per_epoch=1000);history=[]
    for step in range(32):
        start=time.perf_counter();record=update(model,ds,rows,resolver,opt,scheduler,state,runtime);history.append({'step':step+1,'loss':record['loss'],'seconds':time.perf_counter()-start})
        if step%8==7:print('FIXED_A_LEARNING',step+1,record['loss'],flush=True)
    after=measure();a=sum(x['A_mse'] for x in before)/len(before);b=sum(x['A_mse'] for x in after)/len(after)
    assert b<a,(a,b)
    report={'status':'PASS','identity':identity(config,sets,resolver),'initial_parameter_sha256':initial,'initialization':'fresh seed17, no old checkpoint','formal_updates':0,'engineering_updates':32,'selection':'Four-source native-pixel median TRAIN observations; same A mask at epoch0','before':before,'after':after,'initial_mean_A_loss':a,'final_mean_A_loss':b,'history':history,'limits':'Fixed cohort learning control only; does not prove new experiment converged or all colour bias resolved.'}
    write(out/'report.json',report);print('FIXED_A_LEARNING_PASS',a,b,flush=True)

if __name__=='__main__':main()
