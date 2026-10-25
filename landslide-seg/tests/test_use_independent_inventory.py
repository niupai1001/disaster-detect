import importlib.util
import json
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'scripts/build_research_inventory.py'
spec=importlib.util.spec_from_file_location('use_inventory_test',PATH)
inventory=importlib.util.module_from_spec(spec);spec.loader.exec_module(inventory)


def test_probe_names_do_not_invent_mit_or_detail_branch():
    for mode in ('linear','shallow','fpn','unet'):
        _,name=inventory.naming(dict(model=dict(type='use',use_downstream=mode,use={})))
        assert 'SegFormer' not in name and '原网格细节' not in name
    _,rgb=inventory.naming(dict(model=dict(type='use',use_downstream='rgb3',use={})))
    assert '原RGB预训练MiT首层' in rgb


def test_new_output_roots_include_ssl_and_random_probe_without_false_pretraining(tmp_path,monkeypatch):
    monkeypatch.setattr(inventory,'ROOT',tmp_path)
    probe=tmp_path/'outputs/use_independent_frozen/probes/version/random/1234'
    probe.mkdir(parents=True)
    config=dict(segformer=dict(experiment='use_independent_probe_linear_random',seed=1234,
        model=dict(type='use',use_downstream='linear',freeze_use=True,use_stage='independent_random',use_pretrained='initialization.pt'),
        data=dict(manifest='manifests/v4r1/pool_100.jsonl'),train=dict(freeze_bn_running_stats=True),loss={}))
    (probe/'resolved_config.json').write_text(json.dumps(config))
    ssl=tmp_path/'outputs/use_independent_frozen/pretraining/physical';ssl.mkdir(parents=True)
    (ssl/'resolved_config.json').write_text(json.dumps(dict(use_config={},seed=1234,variant='physical',members_sha256='members',
        train_ids=dict(s2=['a','b']),validation_ids=dict(s2=['c']))))
    records,_=inventory.inventory();assert len(records)==2
    random=next(row for row in records if row['legacy_experiment']=='use_independent_probe_linear_random')
    assert 'USE随机初始化' in random['treatment'] and '重建初始化' not in random['treatment']
    assert 'BN' not in random['treatment']
    assert random['effective_structure']['stem'] is False
    pretraining=next(row for row in records if row['model_config']['type']=='use_ssl')
    assert pretraining['n_train']==2 and pretraining['n_val']==1
    assert pretraining['protocol']=='USE_INDEPENDENT_SSL'
    assert pretraining['selection_metric']=='equal_sensor_unlabeled_standardized_mbm_mae'


def test_added_unet_is_indexed_as_random_task_consumer_without_mit_or_bn(tmp_path,monkeypatch):
    monkeypatch.setattr(inventory,'ROOT',tmp_path)
    run=tmp_path/'outputs/use_independent_frozen/unet_control/version/use_independent_unet/1234'
    run.mkdir(parents=True)
    config=dict(segformer=dict(experiment='use_independent_unet',seed=1234,
        model=dict(type='use',use_downstream='unet',freeze_use=True,use_stage='independent_ssl'),
        data=dict(manifest='manifests/v4r1/pool_100.jsonl'),train=dict(freeze_bn_running_stats=True),loss={}))
    (run/'resolved_config.json').write_text(json.dumps(config))
    records,_=inventory.inventory();assert len(records)==1
    row=records[0]
    assert '可训练U-Net' in row['display_name'] and 'SegFormer' not in row['display_name']
    assert 'BN运行统计' not in row['treatment']
    assert row['effective_structure']['stem'] is False
