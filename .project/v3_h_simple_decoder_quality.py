"""CPU-only read-only recheck and physical-scale images of the engineering fit."""
import ssl,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,re
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'earth-representation';sys.path.insert(0,str(ROOT/'src'))
from encoder_v3.hierarchical.train import load_data,build_model,sample_from_dataset,identity,write
from encoder_v3.latent_response.checkpoint import parameter_sha


def main():
    torch.set_num_threads(2);torch.manual_seed(17);torch.use_deterministic_algorithms(True)
    folder=ROOT/'outputs/encoder_v3/hierarchical/simple_decoder';source=folder/'learning_v4';out=folder/(sys.argv[1] if len(sys.argv)>1 else 'quality_final_v1');out.mkdir(exist_ok=False)
    report=json.loads((source/'report.json').read_text());assert report['status']=='PASS'
    config=json.loads((ROOT/'configs/config_v3_h_simple_decoder_v1.json').read_text());sets,resolver=load_data(config)
    model=build_model(config);assert parameter_sha(model)==report['initial_parameter_sha256']
    payload=torch.load(source/'engineering.pt',map_location='cpu',weights_only=False)
    binding=identity(config,sets,resolver)
    for key in ('data','members','physical_catalog','config','architecture'):assert payload['identity'][key]==binding[key]
    model.load_state_dict(payload['model'],strict=True);model.eval();before=parameter_sha(model)
    sha=hashlib.sha256((source/'engineering.pt').read_bytes()).hexdigest();ds=sets['TRAIN'];lookup={r['id']:i for i,r in enumerate(ds.rows)}
    plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,5,figsize=(16,7),layout='constrained');rows=[];picture_row=0
    with torch.no_grad():
        for j,case in enumerate(report['cohort']):
            index=lookup[case['id']];row=ds.rows[index];sample=sample_from_dataset(ds,index,resolver,'cpu')
            loss,d=model([sample],kind='A',seed=17,return_predictions=True);detail=d[0]
            assert abs(float(loss)-report['final_losses'][j])<max(1e-6,abs(report['final_losses'][j])*.002)
            assert float(loss)<=detail['baseline_mae']*1.01+1e-7
            rows.append({'id':case['id'],'source':case['source'],'loss':float(loss),'baseline':detail['baseline_mae']})
            if case['source']!='L9' and case['id']!='obs_9f912ff58940f9eb94093fd1':continue
            head={k:v.clone() for k,v in model.decoder.output.state_dict().items()};model.decoder.output.weight.zero_();model.decoder.output.bias.zero_()
            baseline=model([sample],kind='A',seed=17,return_predictions=True)[1][0];model.decoder.output.load_state_dict(head)
            canonical=lambda s:re.sub(r'^B0+(\d)',r'B\1',s.removeprefix('SR_'))
            names={canonical(c.name):i for i,c in enumerate(sample.channels)};ids=[names[b] for b in ('B4','B3','B2')]
            bank=ds.datasets[row['sensor']=='EnMAP'];truth=[];prediction=[];interpolation=[];hidden=[];valid=[]
            for i in ids:
                c=sample.channels[i];stat=bank.statistics['bands'][row['instrument']+':'+c.name]
                truth.append(c.values.numpy()*stat['std']+stat['mean']);prediction.append(detail['predictions'][i].numpy()*stat['std']+stat['mean'])
                interpolation.append(baseline['predictions'][i].numpy()*stat['std']+stat['mean']);hidden.append(detail['target_masks'][i].numpy());valid.append(c.valid.numpy())
            truth,prediction,interpolation=np.stack(truth,-1),np.stack(prediction,-1),np.stack(interpolation,-1)
            hidden=np.stack(hidden,-1).all(-1);valid=np.stack(valid,-1).all(-1)
            display=lambda a:np.clip(a/.3,0,1)**(1/2.2)
            masked=display(truth);masked[hidden|~valid]=.5
            base=np.where(hidden[...,None],interpolation,truth);learned=np.where(hidden[...,None],prediction,truth)
            for col,(array,title) in enumerate(((display(truth),'原图'),(masked,'可见输入'),(display(base),'原生IDW8基线'),(display(learned),'学习残差后'))):
                axes[picture_row,col].imshow(array);axes[picture_row,col].set_title(title);axes[picture_row,col].axis('off')
            error=np.ma.masked_where(~hidden,np.abs(prediction-truth).mean(-1))
            im=axes[picture_row,4].imshow(error,vmin=0,vmax=.05,cmap='magma');axes[picture_row,4].set_title('隐藏区物理MAE（0–0.05）');axes[picture_row,4].axis('off');fig.colorbar(im,ax=axes[picture_row,4],shrink=.75,extend='max')
            axes[picture_row,0].text(0,-.08,f'{row["sensor"]} TRAIN：基线{detail["baseline_mae"]:.6f} → {float(loss):.6f}',transform=axes[picture_row,0].transAxes,fontsize=10)
            picture_row+=1
    assert picture_row==2 and parameter_sha(model)==before and hashlib.sha256((source/'engineering.pt').read_bytes()).hexdigest()==sha
    fig.suptitle('简洁Decoder工程检查：统一反射率0–0.3 / gamma2.2；拼合图可见部分为原图\n仅5个固定TRAIN观测的工程拟合，未使用正式训练或独立VAL作为拟合目标',fontsize=12)
    fig.savefig(out/'reconstruction.png',dpi=150);plt.close(fig)
    write(out/'report.json',{'status':'PASS','identity':binding,'source_identity':payload['identity'],'members':rows,'parameters_unchanged':True,'engineering_checkpoint_sha256':sha,'gpu_used':False,'optimizer_updates':0,'source_weights_rechecked_on_final_code':True})
    print('FINAL_QUALITY_RECHECK_PASS',rows,flush=True)


if __name__=='__main__':main()
