import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('agent_work', Path(__file__).resolve().parents[1] / 'scripts/manage_agent_work.py')
work = importlib.util.module_from_spec(spec)
spec.loader.exec_module(work)


class WorkRegisterTest(unittest.TestCase):
    def test_explicit_transfer(self):
        temp_root = Path(__file__).resolve().parents[1] / '.tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / 'work.json'
            path.write_text('{"tasks": []}', encoding='utf-8')
            work.update(path, 'claim', 'one', 'agent1', 'source', ['models', 'outputs/old'])
            work.update(path, 'claim', 'two', 'agent2', 'target', ['tests/new'])
            work.update(path, 'note', 'two', 'agent2', evidence=['tests/new'], note='Implementation in progress')
            self.assertEqual(json.loads(path.read_text())['tasks'][1]['status'],'active')
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                work.transfer(path, 'two', 'agent2', 'one', ['models'], '')
            self.assertEqual(before, path.read_bytes())
            work.transfer(path, 'two', 'agent2', 'one', ['models'], 'human authorized')
            data = json.loads(path.read_text())
            self.assertEqual(data['tasks'][0]['scopes'], ['outputs/old'])
            self.assertIn('models', data['tasks'][1]['scopes'])
            self.assertEqual(data['transfers'][0]['authorization'], 'human authorized')

    def test_duplicate_scope_owner_and_lock(self):
        temp_root = Path(__file__).resolve().parents[1] / '.tmp'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            path = Path(directory) / 'work.json'
            path.write_text('{"tasks": []}', encoding='utf-8')
            work.update(path, 'claim', 'one', 'agent1', '文档', ['docs'])
            before = path.read_bytes()
            for scope in ['../docs', 'C:/docs', '/docs', '.']:
                with self.assertRaises(ValueError):
                    work.update(path, 'claim', 'bad', 'agent2', '非法路径', [scope])
                self.assertEqual(before, path.read_bytes())
            for task, owner, scope in [('one', 'agent1', ['src']), ('two', 'agent2', ['DOCS/a.md'])]:
                with self.assertRaises(ValueError):
                    work.update(path, 'claim', task, owner, '重复', scope)
                self.assertEqual(before, path.read_bytes())
            with self.assertRaises(ValueError):
                work.update(path, 'finish', 'one', 'agent2', evidence=['docs/a.md'], note='完成')
            with work.locked(path):
                with self.assertRaises(ValueError):
                    work.update(path, 'claim', 'two', 'agent2', '其他', ['src'])
            work.update(path, 'finish', 'one', 'agent1', evidence=['docs/a.md'], note='完成')
            work.update(path, 'claim', 'two', 'agent2', '其他', ['docs/a.md'])
            self.assertEqual(['completed', 'active'], [t['status'] for t in json.loads(path.read_text())['tasks']])


if __name__ == '__main__':
    unittest.main()
