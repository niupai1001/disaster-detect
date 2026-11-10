import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from encoder_v3.latent_response.enmap_data import build_release,NativeEnmapDataset,KEY
from encoder_v3.latent_response.registry import ChannelCatalog


class EnmapLoaderTest(unittest.TestCase):
    def test_qa_gain_empty_channels_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir();out=root/'release'
            prefix='ENMAP01-____L2A-DT0000000001_20200101T000000Z_001_V010000_20200102T000000Z'
            def tif(suffix,data):
                with rasterio.open(raw/(prefix+suffix),'w',driver='GTiff',height=8,width=8,count=len(data),dtype=str(data.dtype),crs='EPSG:32631',transform=from_origin(0,240,30,30),nodata=-32768 if data.dtype==np.int16 else None) as ds:ds.write(data)
            image=np.tile(np.arange(64,dtype=np.int16).reshape(1,8,8)+1000,(224,1,1));image[130:135]=-32768
            tif('-SPECTRAL_IMAGE_COG.TIF',image)
            masks=np.zeros((224,8,8),dtype=np.uint8);masks[:,0,0]=1;tif('-QL_PIXELMASK_COG.TIF',masks)
            for key in ('CLASSES','CLOUD','CLOUDSHADOW','HAZE','CIRRUS','SNOW','TESTFLAGS'):
                q=np.ones((1,8,8),dtype=np.uint8) if key=='CLASSES' else np.zeros((1,8,8),dtype=np.uint8)
                if key=='CLOUD':q[:,1,1]=1
                tif('-QL_QUALITY_'+key+'_COG.TIF',q)
            xml='<root><processingLevel>L2A</processingLevel>'+''.join(f'<bandID number="{i+1}"><wavelengthCenterOfBand>{400+i}</wavelengthCenterOfBand><FWHMOfBand>7</FWHMOfBand><GainOfBand>0.0001</GainOfBand><OffsetOfBand>0</OffsetOfBand></bandID>' for i in range(224))+'</root>'
            (raw/(prefix+'-METADATA.XML')).write_text(xml)
            doc=build_release(raw,out,[]);self.assertEqual(len(doc['empty_band_ids']),5)
            catalogs={KEY:ChannelCatalog.from_ordered([f'B{i:03d}' for i in range(1,225)])}
            data=NativeEnmapDataset(out,catalogs);observation=data[0];channels=data.channels(observation,'cpu')
            observation.bands[0].reflectance.fill_(999)
            again=data[0];self.assertLess(float(again.bands[0].reflectance.max()),1.)
            self.assertEqual(data.input_cache_hits,1)
            observation=again
            self.assertEqual(len(channels),224);self.assertEqual(channels[130].channel_meta.canonical_ordinal,130)
            self.assertFalse(channels[130].valid.any());self.assertFalse(channels[0].valid[0,0]);self.assertFalse(channels[0].valid[1,1])
            self.assertAlmostEqual(float(observation.bands[0].reflectance[2,2]),.1018,places=6)
            (out/'statistics.json').write_text('{}')
            with self.assertRaises(ValueError):NativeEnmapDataset(out,catalogs)
