import unittest,tempfile,json,hashlib
from pathlib import Path
from .reconcile_discovery import reconcile


class DiscoveryReconcileChecks(unittest.TestCase):
    def test_case_variants_retained_and_exact_paths_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'names.nul';raw.write_bytes(b'C:\\data\\S2B_scene.TIF\0C:\\data\\S2B_scene.TIF\0C:\\data\\S2B_scene.tif\0D:\\unknown.nc\0')
            summary=root/'summary.json';summary.write_text(json.dumps(dict(status='DONE',roots=[dict(root='C:\\',status='FINISHED',filename_list=str(raw),filename_list_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())])))
            result=reconcile([summary]);self.assertEqual(result['unique_exact_normalized_paths'],3);self.assertEqual(result['casefold_alias_collisions_retained'],1);self.assertEqual(result['counts']['optical_name_hint_not_admitted'],2)
            raw.write_bytes(raw.read_bytes()+b'changed')
            with self.assertRaises(ValueError):reconcile([summary])

    def test_live_refinement_is_not_completed_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'summary.json';p.write_text(json.dumps(dict(status='REFINING_NAMES_ONLY',roots=[])))
            with self.assertRaisesRegex(ValueError,'still running'):reconcile([p])

    def test_existing_source_identity_join_keeps_case_aliases_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);raw=root/'names.nul';raw.write_bytes(b'N:\\sources\\BLUE.TIF\0')
            summary=root/'summary.json';summary.write_text(json.dumps(dict(status='DONE',roots=[dict(root='N:\\',status='FINISHED',filename_list=str(raw),filename_list_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())])))
            known=root/'known.json';known.write_text(json.dumps({'N:\\sources\\blue.tif':{'sha256':'a'*64},'N:\\missing.tif':{'sha256':'b'*64}}))
            joined=reconcile([summary],known)['known_source_name_join'];self.assertEqual(joined['exact_matches'],0);self.assertEqual(joined['additional_case_variant_matches'],1);self.assertEqual(joined['not_found_paths'],['N:\\missing.tif'])
