"""Decode a single tile from the EnMAP COG locally, to validate approach C."""

import glob

import numpy as np
import tifffile

f = glob.glob(r"N:\Users\yzh\Disater\earth-representation\data\enmap\*SPECTRAL*.TIF")[0]
tf = tifffile.TiffFile(f)
page = tf.pages[0]
print(f"shape {page.shape} tiles {page.tilewidth}x{page.tilelength} "
      f"tiles_across {page.chunked} segments {len(page.dataoffsets)}")

with open(f, "rb") as fh:
    off, cnt = page.dataoffsets[0], page.databytecounts[0]
    fh.seek(off)
    raw = fh.read(cnt)
print(f"tile0 bytes {cnt/1e6:.1f} MB")

decoded = page.decode(raw, 0)
arr = decoded[0] if isinstance(decoded, tuple) else decoded
print(f"decoded type {type(decoded)} shape {getattr(arr, 'shape', None)} dtype {getattr(arr, 'dtype', None)}")
print(f"range {int(arr.min())}..{int(arr.max())}")

# geometry: where does tile 0 sit?
scale = page.tags.get("ModelPixelScaleTag")
tie = page.tags.get("ModelTiepointTag")
print("pixel scale:", scale.value if scale else None)
print("tiepoint:", tie.value[:6] if tie else None)
