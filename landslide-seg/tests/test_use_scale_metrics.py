import numpy as np
from landslide_seg.training.segformer_metrics import evaluate_tile_v2, summarize_tiles_v2


def test_use_scale_boundaries_and_matching():
    gt=np.zeros((64,64),dtype=bool)
    for y,n in [(5,4),(15,5),(25,25),(35,26)]:gt[y,5:5+n]=True
    row=evaluate_tile_v2(gt.astype('float32'),gt,np.ones_like(gt))
    bins=summarize_tiles_v2([row])['objects']['use_bins']
    assert [bins[k]['n_objects_evaluable'] for k in ('1_4','5_25','26_plus')]==[1,2,1]
    assert all(v['object_recall_evaluable']==1 for v in bins.values())
