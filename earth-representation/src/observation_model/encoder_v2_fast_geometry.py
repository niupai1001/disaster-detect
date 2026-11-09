"""Candidate: share exact neighbor indices for identical visible native grids."""
import hashlib

from .encoder_v2_fast import FastObservationEncoderV2


class GeometryFastObservationEncoderV2(FastObservationEncoderV2):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.neighbor_cache_stats=dict(hits=0,misses=0)
        cache={};encode=self.encode;native_bank=self.encoder.native_bank;nearest=self.decoder.nearest_native
        def bank_with_geometry(band,grid):
            bank=native_bank(band,grid)
            digest=hashlib.sha256(band.keep().detach().cpu().contiguous().numpy().tobytes()).digest()
            bank.neighbor_geometry_key=(tuple(bank.index_map.shape),tuple(band.transform),band.crs,digest)
            return bank
        def encode_current(images,excluded_bands=()):
            cache.clear()
            return encode(images,excluded_bands)
        def nearest_cached(bank,coordinates,bounds):
            key=(bank.neighbor_geometry_key,id(coordinates),tuple(bounds))
            if key not in cache:
                # Retain the query tensor so Python/CUDA cannot reuse its identity
                # for a different chunk while these cached indices remain alive.
                cache[key]=(coordinates,nearest(bank,coordinates,bounds))
                self.neighbor_cache_stats['misses']+=1
            else:self.neighbor_cache_stats['hits']+=1
            return cache[key][1]
        self.encoder.native_bank=bank_with_geometry
        self.encode=encode_current
        self.decoder.nearest_native=nearest_cached
