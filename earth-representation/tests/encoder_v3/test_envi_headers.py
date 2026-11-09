import unittest
from encoder_v3.audit_envi_headers import parse_header


class EnviHeaderChecks(unittest.TestCase):
    def test_multiline_spectrum_is_metadata_only_and_invalid_waves_are_flagged(self):
        text=b'ENVI\nsamples=2\nlines=3\nbands=3\nwavelength={350,\n354,358}\nfwhm={4,4,4}\nbbl={1,1,1}\n'
        row=parse_header(text)
        self.assertEqual(row['wavelength_count'],3)
        self.assertTrue(row['wavelength_count_matches_bands'])
        self.assertTrue(row['wavelengths_increasing'])
        self.assertNotIn('srf',row)
        self.assertFalse(parse_header(text.replace(b'354',b'nan'))['wavelengths_finite'])
        self.assertFalse(parse_header(text.replace(b'354',b'340'))['wavelengths_increasing'])
        self.assertEqual(parse_header(b'not an ENVI header')['status'],'NOT_ENVI_HEADER')
