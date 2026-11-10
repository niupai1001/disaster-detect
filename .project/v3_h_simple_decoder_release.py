"""Freeze a reviewed, identity-bound 30-Epoch release; does not launch training."""
import ssl,os
from pathlib import Path
import argparse,json,subprocess,hashlib,sys,shutil,xml.etree.ElementTree as ET
import torch
ROOT=Path(__file__).resolve().parents[1]/'earth-representation';sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(Path(__file__).parent))
from encoder_v3.hierarchical.train import load_data,build_model,identity,EpochPlan,write
from encoder_v3.hierarchical.verification import fingerprint
from encoder_v3.latent_response.checkpoint import parameter_sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True);p.add_argument('--source-tree',required=True);args=p.parse_args()
    repo=ROOT.parent
    assert subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=repo,text=True).strip()==args.source_tree
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    config=json.loads((ROOT/'configs/config_v3_h_simple_decoder_v1.json').read_text())
    assert config['epochs']==30 and config['warmup_steps']==50 and config['exclude_enmap'] is False
    assert config['task_weights']=={'A':1.,'B1':0.} and config['lambda_jepa']==0 and config.get('max_updates') is None
    sets,resolver=load_data(config);binding=identity(config,sets,resolver);evidence=ROOT/'outputs/encoder_v3/hierarchical/simple_decoder'
    gates={}
    for name,file in {'pipeline':'pipeline_v1/report.json','precision':'precision_final_v2/report.json','quality':'quality_final_v2/report.json','review':'independent_review.json'}.items():
        result=json.loads((evidence/file).read_text());assert result['status']=='PASS' and result['identity']==binding,(name,'gate identity mismatch')
        gates[name]={'path':str((evidence/file).resolve()),'sha256':hashlib.sha256((evidence/file).read_bytes()).hexdigest()}
    suite=ET.parse(evidence/'cpu_final_v1.xml').getroot()[0];assert int(suite.attrib['tests'])>=72 and suite.attrib['failures']==suite.attrib['errors']=='0'
    gates['cpu']={'path':str((evidence/'cpu_final_v1.xml').resolve()),'tests':int(suite.attrib['tests'])}
    plan=EpochPlan(sets['TRAIN'].rows,17,32);assert len(plan)==233 and len(plan.batch(232))==18
    coverage=[r['id'] for i in range(len(plan)) for r in plan.batch(i)];assert len(coverage)==len(set(coverage))==7442
    counts={s:sum(r['sensor']==s for r in sets['TRAIN'].rows) for s in ('S2','L8','L9','EnMAP')};assert counts=={'S2':5349,'L8':1522,'L9':524,'EnMAP':47}
    release=(ROOT/'outputs/encoder_v3/hierarchical/release_simple_decoder_v1').resolve();runtime=release/'runtime';runtime.mkdir(parents=True,exist_ok=False)
    paths=subprocess.check_output(['git','ls-files','src/encoder_v3','configs','scripts/run_v3_h.py','scripts/visualize_v3_h.py','tests/encoder_v3/hierarchical','tests/encoder_v3/shared_mae'],cwd=ROOT,text=True).splitlines()
    manifest={}
    for name in paths:
        src=ROOT/name;dst=runtime/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        manifest[name]=hashlib.sha256(dst.read_bytes()).hexdigest();assert manifest[name]==hashlib.sha256(src.read_bytes()).hexdigest()
    for name in ('data','outputs'):
        subprocess.run(['cmd','/c','mklink','/J',str(runtime/name),str((ROOT/name).resolve())],check=True,capture_output=True)
    contract=release/'contract';contract.mkdir()
    write(contract/'coverage.json',{'TRAIN':7442,'VAL':383,'sources':counts,'batches_per_epoch':233,'tail':18,'epochs':30,'total_updates':6990,'unique_epoch_coverage':True,'warmup_updates':50,'old_runs_resumed':False})
    write(release/'identity.json',binding)
    write(release/'source_manifest.json',{'status':'FROZEN','source_git_commit':args.source_commit,'source_git_tree':args.source_tree,'local_equivalent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'files':manifest})
    torch.manual_seed(17);model=build_model(config);initial=parameter_sha(model);encoder=fingerprint(model.online.state_dict())
    assert initial==json.loads((evidence/'pipeline_v1/report.json').read_text())['initial_parameter_sha256']
    from v3_h_simple_decoder_gate import original_model
    torch.manual_seed(17);original=original_model(config);assert fingerprint(original.online.state_dict())==encoder
    write(release/'initialization.json',{'initial_parameter_sha256':initial,'initial_encoder_sha256':encoder,'random_seed':17,'loaded_checkpoint':None,'original_physical_encoder_initialization_unchanged':True,'decoder_parameters':sum(p.numel() for p in model.decoder.parameters()),'engineering_weights_discarded':True})
    write(release/'readiness.json',{'identity':binding,'training_release_ready':True,'gates':gates,'quality_scope':'5 TRAIN engineering observations; not a claim that 30Epoch or held-out generalization has completed. Exact legal IDW initial baseline and actual learned residual gains checked separately.','precision':'FP32 states/native geometry; gated TF32 GEMM; BF16 rejected','data_sources':counts})
    write(release/'authorization.json',{'source':'user_in_current_chat','identity':binding,'instruction':'完成简洁Decoder、质量及计算加速验收后，保留EnMAP，A-only随机初始化双卡训练30Epoch，预热前50步；排查L9。','epochs':30,'total_updates':6990,'warmup_steps':50,'resume':False})
    driver='import os\nos.environ["CUDA_LAUNCH_BLOCKING"]="0"\nos.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"\nimport ssl,sys,runpy\nfrom pathlib import Path\nentry=Path(__file__).resolve().parent/"runtime/scripts/run_v3_h.py"\nsys.argv=[str(entry),*sys.argv[1:]]\nrunpy.run_path(str(entry),run_name="__main__")\n'
    (release/'run.py').write_text(driver,encoding='utf-8')
    # The frozen data reader must bind the exact same copied bytes, not the working tree.
    code='import ssl,sys,json;from pathlib import Path;sys.path.insert(0,"src");from encoder_v3.hierarchical.train import load_data,identity;c=json.load(open("configs/config_v3_h_simple_decoder_v1.json"));d,r=load_data(c);print(json.dumps(identity(c,d,r)))'
    frozen=json.loads(subprocess.check_output([sys.executable,'-I','-c',code],cwd=runtime,text=True));assert frozen==binding
    print('SIMPLE_RELEASE_FROZEN',release.resolve(),'files',len(manifest),'identity',binding['code'],'initial',initial,flush=True)


if __name__=='__main__':main()
