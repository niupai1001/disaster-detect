"""Matched locations/views and independent adaptive stopping for three routes."""
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs/verification"
def worker(gpu,routes):
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);results=[]
    for route in routes:
        tag="sensor_"+route
        command=[sys.executable,"-u","-X","utf8","-m","scripts.train_dino_official","--adaptive","--route",route,"--sensor-release",str(OUT/"sensor_release"),"--batch-size","32","--steps","50000","--max-epochs","400","--seed","0","--tag",tag,"--out",str(OUT/"runs"),"--device","cuda","--tv-root",str(OUT/"tvlp")]
        status=OUT/("sensor_gpu"+str(gpu)+"_status.json")
        status.write_text(json.dumps(dict(status="running",current=tag,command=command,results=results),indent=2))
        with (OUT/(tag+".console.log")).open("a",encoding="utf-8") as log:
            code=subprocess.run(command,cwd=OUT/"sensor_code_release/earth-representation",env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        results.append(dict(tag=tag,returncode=code))
        if code:break
    status.write_text(json.dumps(dict(status="training_complete_pending_acceptance" if all(r["returncode"]==0 for r in results) else "failed",results=results),indent=2))
    return results

def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,0,["single","paired"]),pool.submit(worker,1,["mixed"])]
        results=[f.result() for f in futures]
    (OUT/"sensor_queue_status.json").write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()
