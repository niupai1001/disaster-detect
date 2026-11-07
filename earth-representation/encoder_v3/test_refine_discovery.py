import json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .refine_discovery import children_for_refinement,main
from .data import sha256


class RefinementChecks(unittest.TestCase):
    def test_only_directory_children_and_stable_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'z').mkdir();(root/'a').mkdir();(root/'image.tif').write_bytes(b'not opened')
            work,skipped=children_for_refinement(root)
            self.assertEqual(work,[str(root/'a'),str(root/'z')]);self.assertEqual(skipped,[])
            with self.assertRaises(TimeoutError):children_for_refinement(root,deadline=time.perf_counter()-1)

    def test_links_and_junctions_are_not_followed(self):
        def entry(name,link=False,directory=True):return SimpleNamespace(path=str(Path('C:/fixture')/name),is_symlink=lambda:link,is_dir=lambda **kwargs:directory)
        with patch('encoder_v3.refine_discovery.os.scandir') as scan,patch.object(Path,'is_junction',lambda p:p.name=='junction'):
            scan.return_value.__enter__.return_value=iter([entry('link',True),entry('junction'),entry('real'),entry('file',directory=False)])
            work,skipped=children_for_refinement('C:/fixture')
        self.assertEqual(work,[str(Path('C:/fixture/real'))]);self.assertEqual({Path(row['root']).name for row in skipped},{'link','junction'})

    def test_budget_stop_preserves_pending_without_false_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'input';target.mkdir();output=root/'audit'
            with patch('encoder_v3.refine_discovery.scan',side_effect=AssertionError('No scan after deadline')):
                main(SimpleNamespace(targets=[str(target)],output_dir=output,max_seconds=1e-12))
            report=json.loads((output/'summary.json').read_text())
            self.assertEqual(report['status'],'FILENAME_REFINEMENT_STOPPED_LIMIT_NOT_ADMISSION');self.assertEqual(report['pending_chunks'],1);self.assertEqual(report['pending_work'][0]['root'],str(target.resolve()));self.assertEqual(report['admitted_members'],0)

    def test_invalid_budget_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'audit'
            for value in (0.,-1.,float('nan')):
                with self.assertRaises(ValueError):main(SimpleNamespace(targets=[str(root)],output_dir=output,max_seconds=value))
            self.assertFalse(output.exists())

    def test_pending_resume_preserves_depth_and_parent_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'input';child=target/'child';child.mkdir(parents=True);parent=root/'parent.json';output=root/'new-output'
            payload=dict(status='FILENAME_REFINEMENT_STOPPED_LIMIT_NOT_ADMISSION',targets=[str(target)],pending_chunks=1,pending_work=[dict(root=str(child),depth=2,max_depth=1)])
            parent.write_text(json.dumps(payload));original=parent.read_bytes()
            def fake_scan(path,directory,**kwargs):
                self.assertEqual(path,str(child.resolve()));self.assertEqual(kwargs['max_depth'],1)
                names=directory/'names.nul';names.write_bytes(b'')
                return dict(root=path,status='FINISHED',files=0,filename_list=str(names),filename_list_sha256=sha256(names))
            with patch('encoder_v3.refine_discovery.scan',side_effect=fake_scan):main(SimpleNamespace(pending_from=parent,targets=None,output_dir=output,max_seconds=10.))
            report=json.loads((output/'summary.json').read_text());self.assertEqual(parent.read_bytes(),original);self.assertEqual(report['status'],'FILENAME_REFINEMENT_FINISHED_NOT_ADMISSION');self.assertEqual(report['roots'][0]['refinement_depth'],2);self.assertEqual(report['roots'][0]['max_depth'],1);self.assertEqual(report['parent_pending_summary']['sha256'],sha256(parent));self.assertEqual(report['pending_chunks'],0)

    def test_pending_resume_rejects_active_scope_escape_and_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);target=root/'input';target.mkdir();parent=root/'parent.json';output=root/'out'
            payload=dict(status='REFINING_NAMES_ONLY',targets=[str(target)],pending_chunks=1,pending_work=[dict(root=str(target),depth=1,max_depth=1)])
            args=SimpleNamespace(pending_from=parent,targets=None,output_dir=output,max_seconds=10.)
            parent.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'terminal'):main(args)
            payload['status']='FILENAME_REFINEMENT_STOPPED_LIMIT_NOT_ADMISSION';payload['pending_work'][0]['root']=str(root/'outside');parent.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,'outside original'):main(args)
            payload['pending_work'][0]['root']=str(target);parent.write_text(json.dumps(payload));output.mkdir()
            with self.assertRaisesRegex(ValueError,'new output directory'):main(args)
