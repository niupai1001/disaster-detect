import shutil
import tempfile
from pathlib import Path
import unittest
from encoder_v3.discover_rasters import scan


class DiscoveryChecks(unittest.TestCase):
    @unittest.skipUnless(shutil.which('rg'), 'ripgrep binary is required by the read-only filename discovery')
    def test_names_include_hidden_and_ignored_files_without_opening_images(self):
        with tempfile.TemporaryDirectory(dir=(Path(__file__).resolve().parents[2] / 'src/encoder_v3')) as tmp:
            root=Path(tmp);source=root/'input';out=root/'output';source.mkdir();out.mkdir()
            hidden=source/'.hidden';hidden.mkdir()
            (hidden/'native.TIF').write_bytes(b'not a readable image')
            (source/'package.zip').write_bytes(b'not a readable archive')
            (source/'.gitignore').write_text('*.tif\n')
            (source/'unrelated.txt').write_text('ignored')
            report=scan(str(source),out)
            self.assertEqual(report['status'],'FINISHED')
            self.assertEqual(report['extensions'],{'.zip':1,'.tif':1})
            self.assertEqual(report['files'],2)
            shallow=root/'shallow';shallow.mkdir()
            report=scan(str(source),shallow,max_depth=1)
            self.assertEqual(report['extensions'],{'.zip':1})
