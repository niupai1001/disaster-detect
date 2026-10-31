import numpy as np

from src.data.adapters._base import scale_reflectance


def test_hls_default_scale():
    dn = np.array([[0, 5000, 10000]], dtype=np.int16)
    out = scale_reflectance("hls", "B04", dn, {"hls_scale": 0.0001, "hls_offset": 0.0})
    assert np.allclose(out, [[0.0, 0.5, 1.0]], atol=1e-6)


def test_hls_offset_from_product_xml():
    # the offset is read per product, not hard-coded
    dn = np.array([[5000]], dtype=np.int16)
    out = scale_reflectance("hls", "B04", dn, {"hls_scale": 0.0001, "hls_offset": 0.25})
    assert np.allclose(out, [[0.75]], atol=1e-6)


def test_hls_fill_value_is_kept_visible():
    dn = np.array([[-9999]], dtype=np.int16)
    out = scale_reflectance("hls", "B04", dn, {"hls_scale": 0.0001, "hls_offset": 0.0})
    assert out[0, 0] < -0.9  # not silently clipped to 0
