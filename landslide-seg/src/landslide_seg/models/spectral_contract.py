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
