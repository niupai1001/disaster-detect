"""Resume the sole G5 run with verified source microbatching."""
from functools import partial

from . import g5_train,g5_fast_train
from .encoder_v2_microbatch import BatchedObservationEncoderV2,batch_backward
from .encoder_v2_runtime import LOG_ROOT
from .g5_protocol import G5
from .expansion_data import ROOT,read,save,digest


def main():
    manifest_path=G5/'acceleration/microbatch_manifest.json';manifest=read(manifest_path)
    for path,sha in manifest['sha256'].items():assert digest(ROOT/path)==sha,path
    check=read(G5/'acceleration/microbatch_check.json');assert check['status']=='PASS' and check['speedup']>1
    microbatch=check['microbatch'];assert microbatch in [2,4]
    revision=dict(name='single_gpu_microbatch',source_microbatch=microbatch,effective_source_batch=32,
        decoder_query_batch_max=microbatch*8192,batched_spectral_fusion=True,batched_spatial_attention=True,
        batched_target_attention=True,scientific_contract_unchanged=True,manifest_sha256=digest(manifest_path))
    original_launch=g5_train.launch
    def launch(resume=False):
        assert resume
        restore=g5_train.restore
        def restore_revision(*args,**kwargs):
            payload=restore(*args,**kwargs);payload['state']['execution_revision']='single_gpu_microbatch'
            payload['state']['execution_manifest_sha256']=digest(manifest_path);payload['state']['execution_microbatch']=microbatch
            return payload
        g5_train.restore=restore_revision
        for run_id in [read(G5/'config.json')['run_id'],'encoder_v2_G5_live_view']:
            path=LOG_ROOT/run_id/'run.json';metadata=read(path);metadata['config']['execution_revision']=revision;save(path,metadata)
        return original_launch(resume=True)
    g5_train.launch=launch
    g5_train.batch_backward=partial(batch_backward,microbatch=microbatch)
    g5_fast_train.GeometryFastObservationEncoderV2=BatchedObservationEncoderV2
    g5_fast_train.main()


if __name__=='__main__':main()
