"""Georeferenced inference on explicit native 1200m product windows; no network I/O."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from rasterio.transform import Affine

from .model import BandRaster, Representation, TargetQuery, make_model, checkpoint_coordinate_mode
from .native_data import LANDSAT, LS_NAME, S2, optical_from_four_view, read_cached


@dataclass
class ImageProduct:
    bands: list
    metadata: dict

    @classmethod
    def read(cls, path, device="cuda"):
        """Read a native cache plus its sidecar, retaining original product provenance."""
        bands, metadata = read_cached(path, device)
        if metadata.get("processing_domain") != "L2_SR" or metadata.get("footprint_m") != 1200:
            raise ValueError("only audited L2 surface-reflectance native 1200m windows are supported")
        if not metadata.get("product_id") or not metadata.get("date"):
            raise ValueError("product ID and actual acquisition date are required")
        for band in metadata["bands"]:
            if not all(key in band for key in ("transform", "crs", "source", "phase_xy_m")):
                raise ValueError("native geographic grid and source provenance are required")
            transform = Affine(*band["transform"][:6])
            gsd = band["native_gsd_m"]
            if (transform.a, transform.e, transform.b, transform.d) != (gsd, -gsd, 0, 0):
                raise ValueError("native window grid does not match declared pixel support")
        return cls(bands, metadata)

    @classmethod
    def from_four_view(cls, base, site, sensor, bbox, device="cuda"):
        """Read existing original L2 product files using the same accepted calibration reader."""
        arrays, metadata = optical_from_four_view(Path(base), site, sensor, bbox)
        bands = [BandRaster(m["name"], torch.as_tensor(arrays[m["name"]], device=device)[None],
                            torch.as_tensor(arrays[m["name"]+"_valid"], device=device)[None],
                            m["wavelength_nm"], m["bandwidth_nm"], m["native_gsd_m"],
                            tuple(m["phase_xy_m"])) for m in metadata["bands"]]
        return cls(bands, metadata)


@dataclass
class ProductRepresentation(Representation):
    geography: dict


@dataclass
class PredictedObservation:
    value: np.ndarray
    valid: np.ndarray
    target_band: str
    processing_domain: str
    units: str
    crs: str
    transform: tuple
    provenance: dict
    model_estimate: bool = True

    def write_geotiff(self, path):
        import rasterio
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", driver="GTiff", height=self.value.shape[0], width=self.value.shape[1],
                           count=1, dtype="float32", crs=self.crs, transform=Affine(*self.transform),
                           nodata=np.nan, compress="deflate") as dst:
            dst.write(np.where(self.valid, self.value, np.nan).astype(np.float32), 1)
            dst.write_mask(self.valid.astype(np.uint8)*255)
            dst.update_tags(target_band=self.target_band, processing_domain=self.processing_domain,
                            units=self.units, model_estimate="true", provenance=json.dumps(self.provenance))


class ConditionalObservationModel:
    """Product interface around an unchanged, checkpoint-compatible shared neural model."""
    def __init__(self, checkpoint, variant=None, statistics=None, device="cuda"):
        self.device = torch.device(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        variant = variant or state.get("variant") or state.get("resolved_config", {}).get("variant", "transformer")
        self.network = make_model(variant,coordinate_mode=checkpoint_coordinate_mode(state)).to(self.device).eval()
        self.network.load_state_dict(state.get("model", state))
        self.statistics = statistics if statistics is not None else state.get("statistics")
        if self.statistics is None:
            raise ValueError("inference needs the exact TRAIN-only statistics used by this checkpoint")
        self.checkpoint = str(Path(checkpoint).resolve())
        self.step = state.get("step")

    def _product(self, image_product):
        if isinstance(image_product, (str, Path)):
            return ImageProduct.read(image_product, self.device)
        if not isinstance(image_product, ImageProduct):
            raise TypeError("provide a native product window or its cache path, not an anonymous array")
        return image_product

    def _bands(self, product, hidden=()):
        result = []
        for band in product.bands:
            if band.name in hidden:
                continue
            stats = self.statistics[band.name]
            result.append(BandRaster(band.name, (band.value.to(self.device)-stats["mean"])/stats["std"],
                                     band.valid.to(self.device), band.wavelength_nm, band.bandwidth_nm,
                                     band.native_gsd_m, band.phase_xy_m))
        return result

    @torch.inference_mode()
    def encode(self, image_product):
        product = self._product(image_product)
        representation = self.network.encode(self._bands(product))
        return ProductRepresentation(representation.tokens, representation.valid, representation.footprint_m,
                                     representation.band_names, product.metadata)

    @torch.inference_mode()
    def predict(self, image_product, target="S2:B11", output_grid="native", target_product=None):
        product = self._product(image_product)
        visible = self._bands(product, hidden=(target,))
        if not visible:
            raise ValueError("at least one visible non-target band is required")
        descriptors = {"S2:"+name: (wave, gsd) for name, wave, gsd in S2}
        descriptors.update({"L8:"+LS_NAME[name]: (wave, gsd) for name, wave, gsd in LANDSAT})
        if target not in descriptors:
            raise ValueError("target must identify an implemented S2 or Landsat L2 surface-reflectance band")
        wavelength, native_gsd = descriptors[target]
        if output_grid == "native":
            reference = product if target_product is None else self._product(target_product)
            matches = [m for m in reference.metadata["bands"] if m["name"] == target]
            if not matches:
                raise ValueError("cross-sensor native output requires target_product to specify its actual grid")
            grid = matches[0]
            if reference.metadata["crs"] != product.metadata["crs"] or not np.allclose(reference.metadata["bbox"], product.metadata["bbox"], atol=1e-5, rtol=0):
                raise ValueError("target grid must cover the same physical window and CRS")
        elif output_grid == "input":
            grid = min((m for m in product.metadata["bands"] if m["name"] != target), key=lambda m: m["native_gsd_m"])
        else:
            raise ValueError("output_grid must be 'native' or 'input'")
        query = TargetQuery(target.split(":")[0], target, wavelength, None, native_gsd,
                            grid["native_gsd_m"], tuple(grid["phase_xy_m"]))
        prediction, coverage = self.network.decoder(self.network.encode(visible), query)
        return PredictedObservation(prediction[0].float().cpu().numpy(), coverage[0].cpu().numpy(), target,
                                    "L2_SR", "dimensionless_surface_reflectance", grid["crs"],
                                    tuple(grid["transform"][:6]), dict(checkpoint=self.checkpoint, step=self.step,
                                    input_product=product.metadata["product_id"], input_date=product.metadata["date"],
                                    output_grid=output_grid, target_values_used=False,
                                    grid_finer_than_target_instrument=grid["native_gsd_m"] < native_gsd,
                                    coverage="prediction context coverage; not a measured target validity mask"))
