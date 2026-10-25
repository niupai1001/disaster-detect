from pathlib import Path
import runpy
import pytest


def test_next_stage_requires_all_completed_formal_probe_commands():
    path=Path(__file__).resolve().parents[1]/'outputs/use_independent_frozen/run_multisensor_pretraining.py'
    module=runpy.run_path(str(path));ready=module['s2_queue_finished']
    assert not ready(dict(status='running',jobs=[]))
    with pytest.raises(RuntimeError,match='failed'):
        ready(dict(status='failed',jobs=[]))
    with pytest.raises(RuntimeError,match='lacks required'):
        ready(dict(status='finished_requires_result_acceptance',jobs=[]))
    jobs=[dict(label=f'{prefix}_{mode}_{candidate}{suffix}',status='finished',exit_code=0)
          for candidate in module['CANDIDATES'] for mode in ('linear','shallow')
          for prefix,suffix in (('preflight_probe',''),('formal_probe',''),('eval_probe','_best_full'),('eval_probe','_last_full'))]
    assert ready(dict(status='finished_requires_result_acceptance',jobs=jobs))
    jobs[-1]['status']='running'
    with pytest.raises(RuntimeError,match='unresolved child'):
        ready(dict(status='finished_requires_result_acceptance',jobs=jobs))
