import numpy as np

from src.data.adapters._base import scale_reflectance


def test_landsat_c2_formula():
    # the Collection 2 rule: reflectance = DN * 0.0000275 - 0.2
    dn = np.array([[0, 7273, 20000, 40000]], dtype=np.uint16)
    out = scale_reflectance("landsat", "red", dn, {})
    assert np.allclose(out, [[-0.2, 0.0000075, 0.35, 0.9]], atol=1e-6)


def test_landsat_zero_is_not_zero_reflectance():
    # DN 0 is outside the valid range; the offset makes it negative, which the
    # adapter must not clamp silently (QA decides validity instead)
    out = scale_reflectance("landsat", "red", np.array([[0]], dtype=np.uint16), {})
    assert out[0, 0] < 0.0
