from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from use_next_p2.run import evaluate


class Dataset:
    def __len__(self):return 2
    def __getitem__(self,i):
        return dict(image=torch.full((1,4,4),-10.),mask=torch.ones(4,4),valid=torch.ones(4,4,dtype=torch.bool),
                    sample_id=str(i),event_id=('hiroshima','thrissur')[i])


class Model(torch.nn.Module):
    def forward(self,x):return x


def test_full_metric_schema_and_event_mean():
    result=evaluate(Model(),Dataset(),'cpu')
    assert result['event_mean_iou']==0.
    assert set(result['by_event'])=={'hiroshima','thrissur'}
    assert all('pixel' in values and 'pixel_ap' in values for values in result['by_event'].values())
    assert 'pixel_ap' in result and len(result['tiles'])==2
