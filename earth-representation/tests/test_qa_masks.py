import numpy as np

from src.data import qa_masks


def test_s2_scl_classes():
    scl = np.array([[4, 5, 6, 3, 9]], dtype=np.uint8)   # veg, bare, water, shadow, cloud
    masks = qa_masks.s2_scl_masks(scl)
    assert masks["valid"].tolist() == [[True, True, True, False, False]]
    assert masks["cloud_shadow"][0, 3]
    assert masks["cloud_high"][0, 4]


def test_hls_fmask_bits():
    fmask = np.array([[0, 1, 2, 4, 8, 16]], dtype=np.uint8)
    masks = qa_masks.hls_fmask_masks(fmask)
    assert masks["fill"][0, 1]
    assert masks["cloud"][0, 2]
    assert masks["adjacent_cloud"][0, 3]
    assert masks["cloud_shadow"][0, 4]
    assert masks["snow"][0, 5]
    assert masks["valid"].tolist() == [[True, False, False, False, False, False]]


def test_landsat_qa_pixel_and_radsat():
    qa = np.array([[0, 1 << 3, 1 << 4, 1 << 1, 1 << 6]], dtype=np.uint16)
    radsat = np.array([[0, 0, 0, 0, 1]], dtype=np.uint16)
    masks = qa_masks.landsat_qa_masks(qa, radsat)
    assert masks["cloud"][0, 1]
    assert masks["cloud_shadow"][0, 2]
    assert masks["dilated_cloud"][0, 3]
    # bit 6 is "clear" (good), but the radiometric saturation still invalidates it
    assert masks["valid"].tolist() == [[True, False, False, False, False]]


def test_valid_mask_dispatch_covers_all_products():
    assert set(qa_masks.DISPATCH) == {"sentinel2", "hls", "landsat"}
    for product in qa_masks.DISPATCH:
        assert qa_masks.lookalike_rule(product)


def test_constant_qa_yields_constant_mask():
    arr = np.full((4, 4), 6, dtype=np.uint8)       # S2 bare soil
    assert qa_masks.valid_mask("sentinel2", {"SCL": arr}).all()
