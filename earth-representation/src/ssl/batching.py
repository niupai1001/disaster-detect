"""Batching across products: pad to a common band count and carry a mask.

Batches built from one scene make every sample nearly identical, so DINO's
centring removes the only signal there is. Mixing products needs a common
tensor shape, which is what the band mask is for.
"""

import numpy as np
import torch


def pad_bands(image, centre, gsd, n_max):
    """Pad [C,H,W] to n_max bands with zeros and return the availability mask."""
    c = image.shape[0]
    if c > n_max:
        raise ValueError(f"patch has {c} bands, more than {n_max}")
    out = np.zeros((n_max, *image.shape[1:]), dtype=image.dtype)
    out[:c] = image
    mask = np.zeros(n_max, dtype=bool)
    mask[:c] = True
    cpad = np.zeros(n_max, dtype=np.float32)
    cpad[:c] = centre
    gpad = np.ones(n_max, dtype=np.float32)
    gpad[:c] = gsd
    return out, cpad, gpad, mask


def mixed_batches(items, batch_size, rng):
    """Shuffle everything, then pad each batch to its own maximum band count."""
    order = rng.permutation(len(items))
    for i in range(0, len(items), batch_size):
        chunk = [items[j] for j in order[i:i + batch_size]]
        if not chunk:
            continue
        n_max = max(len(it["bands"]) for it in chunk)
        images, centres, gsds, masks = [], [], [], []
        for it in chunk:
            img, cpad, gpad, mask = pad_bands(
                it["image"], np.asarray(it["centre"], np.float32),
                np.asarray(it["gsd"], np.float32), n_max)
            images.append(img)
            centres.append(cpad)
            gsds.append(gpad)
            masks.append(mask)
        yield {
            "image": torch.from_numpy(np.stack(images)).float(),
            "centre": torch.from_numpy(np.stack(centres)),
            "gsd": torch.from_numpy(np.stack(gsds)),
            "mask": torch.from_numpy(np.stack(masks)),
            "bands": chunk[0]["bands"],
        }
