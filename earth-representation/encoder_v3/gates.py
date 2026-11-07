"""Fail closed; a file containing candidates is not a training authorization."""
import argparse
import json
from pathlib import Path


def require_training(config):
    required=config.get('required_before_training')
    if not isinstance(required,dict) or not required:
        raise ValueError('Missing required_before_training')
    schema={'merge_configuration','hypernetwork_hidden_structure','frozen_data_manifest','pair_admission','mask_rectangle_distribution','validation_panel','improvement_threshold','stability_window','early_stop','precision_acceptance','resource_limit','execution_equivalence','optimizer_choice','formal_training_authorization'}
    missing=sorted(schema-set(required))+[key for key,value in required.items() if value is None or value is False or value=='']
    if required.get('formal_training_authorization') is not True and 'formal_training_authorization' not in missing:
        missing.append('formal_training_authorization')
    if missing:
        raise ValueError('Training blocked: '+', '.join(missing))
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    args=parser.parse_args()
    require_training(json.loads(args.config.read_text(encoding='utf-8')))
