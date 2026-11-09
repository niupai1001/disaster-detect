import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from encoder_v3.data import NativeOpticalDataset,sha256


class DataChecks(unittest.TestCase):
    def fixture(self,root,source_type):
        release=root/'release';contracts=root/'contracts';release.mkdir();contracts.mkdir()
        cache=release/'cache.npz';np.savez(cache,B02=np.ones((2,2),dtype=np.float32),B02__valid=np.ones((2,2),dtype=bool))
        srf=contracts/'srf.npz';np.savez(srf,spectral_wavelength_grid_nm=np.array([400.,500.,600.]),normalized_response=np.array([0.,.01,0.]))
        transform=[10.,0.,0.,0.,-10.,20.];grid=dict(shape=[2,2],affine_transform=transform,crs='EPSG:32631')
        row=dict(id='one',role='TRAIN',units='surface_reflectance',cache=str(cache),cache_sha256=sha256(cache),date='2024-01-01',bands=[dict(band_key='B02',shape=[2,2],transform=transform,crs=grid['crs'])])
        (release/'members.jsonl').write_text(json.dumps(row)+'\n')
        contract=dict(row,timestamp=row['date'],values_already_scaled=True,bands=[dict(band_key='B02',grid=grid,spectral_response_ref='curve')])
        (contracts/'observation_contracts.jsonl').write_text(json.dumps(contract)+'\n')
        (contracts/'srf_manifest.json').write_text(json.dumps(dict(curves=[dict(id='curve',source_type=source_type,wavelength_unit='nm',source_url='https://example.invalid/synthetic-test-only',file_hash='0'*64,path=str(srf),sha256=sha256(srf))])))
        manifest_hash=sha256(release/'members.jsonl')
        (release/'release.json').write_text(json.dumps(dict(data_release_accepted=True,manifest_sha256=manifest_hash)))
        (contracts/'data_contract_tests.json').write_text(json.dumps(dict(status='PASS',data_manifest_sha256=manifest_hash,observation_contracts_sha256=sha256(contracts/'observation_contracts.jsonl'),srf_manifest_sha256=sha256(contracts/'srf_manifest.json'))))
        return release,contracts,cache

    def test_supported_srf_and_tamper_detection(self):
        for kind in ('measured_band_averaged_srf','transformed_instrument_rsr'):
            with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
                root=Path(folder).resolve();self.assertTrue(root.is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')))
                release,contracts,cache=self.fixture(root,kind)
                dataset=NativeOpticalDataset(release,contracts,role='TRAIN')
                item=dataset[0];self.assertEqual(item.bands[0].reflectance.shape,(2,2))
                item.bands[0].wavelengths_nm.zero_()
                self.assertGreater(float(dataset[0].bands[0].wavelengths_nm.sum()),0)
                with cache.open('ab') as stream:stream.write(b'changed')
                with self.assertRaises(ValueError):dataset[0]

    def test_fake_srf_rejected(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as folder:
            root=Path(folder).resolve();self.assertTrue(root.is_relative_to((Path(__file__).resolve().parents[2] / 'src/encoder_v3')))
            release,contracts,_=self.fixture(root,'gaussian_approximation')
            with self.assertRaises(ValueError):NativeOpticalDataset(release,contracts,role='TRAIN')[0]
            with self.assertRaises(ValueError):NativeOpticalDataset(release,contracts,role='TEST')
