"""Current-stage authorization; historical frozen sources remain unchanged."""
import json
from pathlib import Path


def require_training_authorization():
    policy=json.loads((Path(__file__).parent/'execution_policy.json').read_text())
    if not policy['formal_training_enabled']:
        raise RuntimeError('Stage restricted to token prototype: formal/whole-band/cross-source/physics training is disabled by user.')
