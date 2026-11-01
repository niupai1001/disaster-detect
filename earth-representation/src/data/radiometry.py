"""Product quantities; decode before interpolating integer measurements."""
import numpy as np

def reflectance(dn, product):
    a = np.asarray(dn, dtype=np.float32)
    if product == "landsat":
        return a * np.float32(0.0000275) - np.float32(0.2)
    if product == "hls":
        return a * np.float32(0.0001)
    raise ValueError(product)
