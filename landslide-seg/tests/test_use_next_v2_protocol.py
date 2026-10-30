import copy
import sys
from pathlib import Path

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_research_protocol import audit_config


def test_native_config_and_budget_override():
    for name in ('dofa','anysat'):
        result=audit_config(ROOT/f'configs/use_next_v2/p1_{name}.yaml')
        assert result['use_next_v2']['ok'] and result['counts']=={'train':2223,'dev':669}
    with pytest.raises(ValueError):
        audit_config(ROOT/'configs/use_next_v2/p1_dofa.yaml',['use_next_v2.schedule_updates=4000'])
