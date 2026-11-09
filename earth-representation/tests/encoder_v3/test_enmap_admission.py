"""EnMAP must remain excluded until complete native calibration/QA/role/statistics admission."""
from pathlib import Path
import tempfile
import unittest
from encoder_v3.enmap_admission import spectral_metadata,native_blockers,overlap


class EnmapAdmissionTests(unittest.TestCase):
    def test_missing_qa_never_passes_with_nodata_only(self):
        self.assertIn('MISSING_PIXEL_QA_CLOUD_SHADOW_CIRRUS_HAZE_SNOW_TESTFLAGS',native_blockers(True,False,True,True))
        self.assertEqual(native_blockers(True,True,True,True),[])
        self.assertEqual(len(native_blockers(False,False,False,False)),4)

    def test_product_gain_and_fixed_channel_ordinal(self):
        xml='<root><processingLevel>L2A</processingLevel><bandID number="1"><wavelengthCenterOfBand>418.416</wavelengthCenterOfBand><FWHMOfBand>7</FWHMOfBand><GainOfBand>0.0001</GainOfBand><OffsetOfBand>0</OffsetOfBand></bandID></root>'
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metadata.xml';p.write_text(xml)
            band=spectral_metadata(p)[0]
            self.assertEqual((band['channel_id'],band['ordinal']),('B001',0))
            self.assertAlmostEqual(1000*band['gain']+band['offset'],.1)
            for bad in (xml.replace('0.0001','-1'),xml.replace('number="1"','number="2"'),xml.replace('L2A','L1B')):
                p.write_text(bad)
                with self.assertRaises(ValueError):spectral_metadata(p)

    def test_bbox_role_overlap_is_conservative(self):
        self.assertTrue(overlap([0,0,3,3],[1,1,4,4]))
        self.assertFalse(overlap([0,0,1,1],[1,0,2,1]))
