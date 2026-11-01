"""One scale correction only, three runs; existing no-scale baseline reused."""
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/verification"
def worker(gpu,modes):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),EARTH_S2_ROOT=str(ROOT.parent/"landslide-seg/data/sen12"),EARTH_MANIFEST_ROOT=str(ROOT.parent/"landslide-seg/manifests"),EARTH_PROTOCOL_MANIFEST=str(ROOT.parent/"landslide-seg/manifests/v4r1/pool_100.jsonl"))
    results=[]
    for mode in modes:
        tag="corrective_scale_"+mode
        command=[sys.executable,"-u","-X","utf8","-m","scripts.train_dino_official","--adaptive","--source","landslide","--n-patches","2223","--batch-size","32","--steps","50000","--max-epochs","400","--seed","0","--scale-mode",mode,"--scale-lambda","0.1","--tag",tag,"--out",str(OUT/"runs"),"--device","cuda","--tv-root",str(OUT/"tvlp")]
        status=OUT/("scale_gpu"+str(gpu)+"_status.json")
        status.write_text(json.dumps(dict(status="running",command=command,current=tag,results=results),indent=2))
        with (OUT/(tag+".console.log")).open("a",encoding="utf-8") as log:
            code=subprocess.run(command,cwd=OUT/"scale_release/earth-representation",env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        results.append(dict(tag=tag,returncode=code))
        if code:break
    status.write_text(json.dumps(dict(status="training_complete_pending_acceptance" if all(r["returncode"]==0 for r in results) else "failed",results=results),indent=2))
    return results

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,0,["uniform","selective"]),pool.submit(worker,1,["random"])]
        results=[f.result() for f in futures]
    (OUT/"scale_queue_status.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()
