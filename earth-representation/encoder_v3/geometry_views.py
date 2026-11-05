"""Exact coordinate reexpression for UTM false-northing conventions; no resampling."""
from dataclasses import replace
from rasterio.crs import CRS
from .data import Observation


def northing_shift(source_crs,target_crs):
    source=CRS.from_user_input(source_crs);target=CRS.from_user_input(target_crs)
    if any(not crs.is_projected or crs.linear_units not in ('metre','meter') for crs in (source,target)):raise ValueError('Metric projected geometry required')
    if source==target:return 0.
    a=source.to_epsg();b=target.to_epsg()
    if not a or not b or a//100 not in (326,327) or b//100 not in (326,327) or a%100!=b%100:raise ValueError('Different native projections require a separate geometry operator; no image warp')
    return (int(b//100==327)-int(a//100==327))*10_000_000.


def common_metric_view(observation,target_crs):
    """Preserve every input value/mask/SRF tensor and record original native grids."""
    target=CRS.from_user_input(target_crs).to_string();bands=[];audit=[]
    for band in observation.bands:
        band.validate();shift=northing_shift(band.crs,target)
        affine=tuple(band.affine[:5])+(band.affine[5]+shift,)
        bands.append(replace(band,affine=affine,crs=target))
        audit.append(dict(band=band.record_id,original_crs=band.crs,original_affine=band.affine,target_crs=target,target_affine=affine,northing_shift_m=shift,resampled=False))
    if all(b.crs==target for b in observation.bands):return observation
    return Observation(bands,dict(observation.metadata,geometry_view=dict(method='UTM_false_northing_only',bands=audit)))
