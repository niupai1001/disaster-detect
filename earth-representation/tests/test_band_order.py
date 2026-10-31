import pytest

from src.data import band_catalog


def test_reordering_keeps_metadata_in_step():
    a = band_catalog.describe("sentinel2", ["B04", "B03"])
    b = band_catalog.describe("sentinel2", ["B03", "B04"])
    assert [m["name"] for m in a] == ["S2_B04", "S2_B03"]
    assert [m["name"] for m in b] == ["S2_B03", "S2_B04"]
    assert a[0]["centre_nm"] == 664.6


def test_unknown_band_is_an_error_not_a_guess():
    with pytest.raises(KeyError):
        band_catalog.stack("sentinel2", ["B13"])


def test_same_wavelength_differs_by_product():
    # B8A and B08 are different bands, and HLS bands keep their own identity
    s2 = {b.key: b for b in band_catalog.BY_PRODUCT["sentinel2"]}
    assert s2["B08"].native_gsd_m == 10
    assert s2["B8A"].native_gsd_m == 20
    hls = {b.key: b for b in band_catalog.BY_PRODUCT["hls"]}
    assert hls["B8A"].name == "HLS_B8A"
    assert hls["B8A"].native_gsd_m == 30


def test_every_product_band_has_unique_name():
    for product, bands in band_catalog.BY_PRODUCT.items():
        names = [b.name for b in bands]
        assert len(names) == len(set(names)), product
