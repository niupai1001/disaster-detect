import numpy as np

from src.data.adapters._base import scale_reflectance


def test_baseline_4_offset_is_subtracted_before_scaling():
    # baseline >= 04.00 stores reflectance as DN + 1000; 1000 DN must be 0.0
    dn = np.array([[1000, 2000, 10000]], dtype=np.uint16)
    out = scale_reflectance("sentinel2", "B04", dn,
                            {"quantification": 10000.0, "boa_offset": -1000.0})
    assert np.allclose(out, [[0.0, 0.1, 0.9]], atol=1e-6)


def test_old_baseline_has_no_offset():
    dn = np.array([[0, 10000]], dtype=np.uint16)
    out = scale_reflectance("sentinel2", "B04", dn,
                            {"quantification": 10000.0, "boa_offset": 0.0})
    assert np.allclose(out, [[0.0, 1.0]], atol=1e-6)


def test_empty_patch_does_not_crash():
    out = scale_reflectance("sentinel2", "B04", np.zeros((0, 0), dtype=np.uint16),
                            {"quantification": 10000.0, "boa_offset": 0.0})
    assert out.shape == (0, 0)
