import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('independent_data', Path(__file__).resolve().parents[1]/'scripts/audit_use_independent_data.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_cache_audit_rejects_tampering_labels_and_invalid_pixels(tmp_path):
    path = tmp_path/'sample.npz'
    arrays = dict(x=np.ones((3, 4, 4), dtype=np.float32), pixel_valid=np.ones((3, 4, 4), dtype=bool),
                  metadata=np.ones((3, 3), dtype=np.float32), known=np.ones((3, 3), dtype=bool))
    row = dict(file='sample.npz', band_ids=['red', 'green', 'blue'], crs='EPSG:32645', transform=[10, 0, 0, 0, -10, 0])
    def write():
        np.savez(path, **arrays)
        row['sha256'] = audit.sha(path)
    write()
    assert audit.audit_cache(tmp_path, row)['common_valid_ratio'] == 1
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        audit.audit_cache(tmp_path, row)
    arrays['label'] = np.zeros((4, 4))
    write()
    with pytest.raises(ValueError, match='fields'):
        audit.audit_cache(tmp_path, row)
    del arrays['label']
    arrays['pixel_valid'][:, :2] = False
    write()
    with pytest.raises(ValueError, match='80%'):
        audit.audit_cache(tmp_path, row)
