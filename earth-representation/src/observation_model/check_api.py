"""One real native product: grid, hidden-target isolation and GeoTIFF round trip."""
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from .api import ConditionalObservationModel, ImageProduct
from .data import ROOT, rows


def main():
    out = ROOT/"observation_model/outputs/phase1/api"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT/"observation_model/outputs/phase1/gpu100/last.pt"
    row = next(r for r in rows("val") if r["sensor"] == "S2")
    product = ImageProduct.read(row["cache"])
    model = ConditionalObservationModel(checkpoint)
    representation = model.encode(product)
    result = model.predict(product, "S2:B11", "native")
    assert result.value.shape == (60, 60) and representation.footprint_m == 1200
    target = next(b for b in product.bands if b.name == "S2:B11")
    target.value.add_(1000)
    repeated = model.predict(product, "S2:B11", "native")
    assert np.array_equal(result.value, repeated.value), "hidden target affected prediction"
    cross = model.predict(product, "L8:SR_B5", "input")
    assert cross.value.shape == (120, 120) and cross.provenance["grid_finer_than_target_instrument"]
    try:
        model.predict(product, "L8:SR_B5", "native")
    except ValueError as exc:
        assert "target_product" in str(exc)
    else:
        raise AssertionError("native Landsat grid cannot be inferred from the S2 grid")
    result.write_geotiff(out/"s2_b11_model_estimate.tif")
    with rasterio.open(out/"s2_b11_model_estimate.tif") as src:
        assert tuple(src.transform)[:6] == result.transform
        assert src.tags()["model_estimate"] == "true"
        assert np.array_equal(src.dataset_mask() > 0, result.valid)
        assert np.allclose(src.read(1)[result.valid], result.value[result.valid])
    evidence = dict(status="passed", checkpoint=str(checkpoint), checkpoint_role="100-step engineering only",
                    product_id=product.metadata["product_id"], hidden_target_max_delta=0,
                    native_output_shape=list(result.value.shape), input_grid_cross_shape=list(cross.value.shape),
                    representation_shape=list(representation.tokens.shape), target_values_used=False,
                    finite=bool(np.isfinite(result.value[result.valid]).all()),
                    boundary="explicit 1200m windows; no automatic full-scene mosaic or undocumented product parsing")
    assert evidence["finite"]
    (out/"acceptance.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence))


if __name__ == "__main__":
    main()
