from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from .audit_lineage import audit_member,quality_mask
from .data import sha256


class LineageChecks(unittest.TestCase):
    def test_parent_calibration_mask_and_tampering(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder);source=root/'source.npz';cache=root/'cache.npz'
            x=np.array([[[.1,.2],[.3,.4]]],np.float32);valid=np.array([[[True,False],[True,True]]])
            np.savez(source,x=x,pixel_valid=valid,metadata=np.array([[.5,0.,10.]]))
            expected=np.where(valid[0],x[0]+np.float32(.1),0).astype(np.float32)
            np.savez(cache,B02=expected,B02__valid=valid[0])
            row=dict(id='fixture',role='TRAIN',source_role='TRAIN',sensor='S2',provenance='native_lattice_cache_channels_only',source_cache=str(source),source_cache_sha256=sha256(source),cache=str(cache),cache_sha256=sha256(cache),calibration_revision=dict(additive_correction=.1),bands=[dict(band_key='B02',source_cache_channel=0,cached_already_scaled=True,shape=[2,2],transform=[10,0,0,0,-10,20],crs='EPSG:32631',native_pixel_support_m=[10,10])])
            result=audit_member(row,None,root)
            self.assertFalse(result['raw_QA_recomputed']);self.assertEqual(result['bands'][0]['max_absolute_difference'],0)
            with self.assertRaisesRegex(ValueError,'Protected role'):audit_member(dict(row,role='TEST'),None,root)
            with source.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'):audit_member(row,None,root)

    def test_native_tif_qa_and_grid(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root=Path(folder);transform=rasterio.Affine(30,0,300,0,-30,600)
            def tif(name,array,grid=transform,nodata=None):
                path=root/(name+'.tif')
                with rasterio.open(path,'w',driver='GTiff',height=2,width=3,count=1,dtype='uint16',crs='EPSG:32631',transform=grid,nodata=nodata) as target:target.write(array.astype(np.uint16),1)
                return path
            dn=np.array([[0,10000,12000],[13000,14000,15000]],np.uint16)
            source=tif('blue',dn,nodata=0)
            pixel=np.zeros((2,3),np.uint16);pixel[0,1]=8;tif('qa_pixel',pixel)
            tif('qa_radsat',np.zeros((2,3)))
            aerosol=np.zeros((2,3),np.uint16);aerosol[1,1]=192;tif('qa_aerosol',aerosol)
            valid=np.array([[False,False,True],[True,False,True]])
            cache=root/'cache.npz';np.savez(cache,SR_B2=np.where(valid,dn.astype(np.float32)*.0000275-.2,0).astype(np.float32),SR_B2__valid=valid)
            row=dict(id='fixture',role='TRAIN',source_role='TRAIN',sensor='L8',provenance='new_native_Landsat_existing_local_anchor',cache=str(cache),cache_sha256=sha256(cache),bands=[dict(band_key='SR_B2',source=str(source),shape=[2,3],transform=list(transform)[:6],crs='EPSG:32631',scale=.0000275,offset=-.2)])
            result=audit_member(row,None,root);self.assertTrue(result['raw_QA_recomputed']);self.assertEqual(result['bands'][0]['valid_pixels'],3)
            tif('qa_aerosol',aerosol,grid=rasterio.Affine(30,0,301,0,-30,600))
            with self.assertRaisesRegex(ValueError,'QA grid'):audit_member(row,None,root)
            with self.assertRaisesRegex(ValueError,'Unknown historical QA'):quality_mask('unknown.tif',aerosol,valid,'L8')
