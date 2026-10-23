"""Named band identities shared by new spectral experiments.

Nominal wavelengths: existing v4r3 TerraTorch 1.2.13 DOFA mapping.
Do not infer band identity from a channel's numerical position.
"""
S2_WAVELENGTH_UM = {
    'B02': .490, 'B03': .560, 'B04': .665, 'B05': .705, 'B06': .740,
    'B07': .783, 'B08': .832, 'B8A': .864, 'B11': 1.610, 'B12': 2.200,
}

def wavelengths_for(bands):
    if len(set(bands)) != len(bands):
        raise ValueError('Duplicate band identities')
    return tuple(S2_WAVELENGTH_UM[b] for b in bands)


RADIOMETRY_MODES = ("legacy_dn", "harmonized_dn_v1")


def harmonized_reflectance(x, mean, std):
    """官方harmonized存储DN/10000；不重复校正日期偏移，负舍入误差截为零。"""
    if x.ndim != 4 or x.shape[1] != 10 or mean.numel() != 10 or std.numel() != 10:
        raise ValueError("反射率契约要求十波段四维输入及十波段统计")
    dn = x * std.reshape(1, 10, 1, 1) + mean.reshape(1, 10, 1, 1)
    return (dn / 10000.0).clamp_min(0.0)


def spectral_indices(rho, eps=1e-4):
    """固定顺序NDVI(B8A/B04)、NDMI、NBR、红边差；eps单位与rho一致。"""
    import torch
    pairs = ((7, 2), (7, 8), (7, 9), (4, 3))
    return torch.cat([(rho[:, a:a+1] - rho[:, b:b+1]) /
                      (rho[:, a:a+1] + rho[:, b:b+1] + eps) for a, b in pairs], dim=1)
