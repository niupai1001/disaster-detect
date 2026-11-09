"""Read-only cross-project claim inspection. Never starts jobs or edits registers."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

SOURCES = (
    ('landslide-seg', 'landslide-seg/docs/assets/research/agent_work.json'),
    ('earth-representation', 'earth-representation/outputs/encoder_v3/agent_work.json'),
)


def normalize(base, scope):
    scope = scope.replace('\\', '/').rstrip('/').casefold()
    p = PurePosixPath(scope)
    if not scope or ':' in scope or p.is_absolute() or '..' in p.parts or scope == '.':
        raise ValueError('invalid relative scope: ' + scope)
    return str(PurePosixPath(base.casefold()) / p)


def overlap(a, b):
    return a == b or a.startswith(b + '/') or b.startswith(a + '/')


def inspect(root):
    result = {'read_only': True, 'runtime_verified': False, 'sources': [],
              'active_or_blocked': [], 'findings': []}
    occupied = []
    for base, relative in SOURCES:
        path = root / relative
        try:
            raw = path.read_bytes()
            data = json.loads(raw.decode('utf-8-sig'))
            tasks = data['tasks']
            if not isinstance(tasks, list):
                raise ValueError('tasks must be an array')
        except (OSError, ValueError, KeyError) as exc:
            result['findings'].append({'source': relative, 'error': str(exc)})
            continue
        result['sources'].append({'path': relative, 'sha256': hashlib.sha256(raw).hexdigest(),
                                  'tasks': len(tasks), 'updated_at': data.get('updated_at')})
        for task in tasks:
            if not isinstance(task, dict):
                result['findings'].append({'source': relative, 'error': 'non-object task'})
                continue
            if task.get('status') not in ('active', 'blocked'):
                continue
            key = relative + '#' + str(task.get('id', '?'))
            result['active_or_blocked'].append({'source': relative, **task})
            if not task.get('scopes'):
                result['findings'].append({'task': key, 'issue': 'empty scopes'})
            for scope in task.get('scopes', []):
                try:
                    name = normalize(base, scope)
                except (ValueError, AttributeError) as exc:
                    result['findings'].append({'task': key, 'issue': str(exc)})
                    continue
                if task['status'] == 'active':
                    for prior, prior_name in occupied:
                        if prior != key and overlap(name, prior_name):
                            result['findings'].append({'task': key, 'other': prior,
                                                       'issue': 'overlapping active scopes', 'scope': name})
                    occupied.append((key, name))
    return result


def self_test():
    assert normalize('earth-representation', r'encoder_v3\reports') == 'earth-representation/encoder_v3/reports'
    assert overlap('a/b', 'a/b/c') and not overlap('a/b', 'a/b2')
    for scope in ('../x', 'C:/x', '/x', '.', ''):
        try:
            normalize('base', scope)
        except ValueError:
            pass
        else:
            raise AssertionError(scope)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / SOURCES[0][1]
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({'tasks': [
            {'id': 'a', 'status': 'active', 'scopes': ['docs']},
            {'id': 'b', 'status': 'active', 'scopes': ['docs/file.md']},
            {'id': 'c', 'status': 'active', 'scopes': []}]}), encoding='utf-8')
        before = target.read_bytes()
        result = inspect(root)
        assert target.read_bytes() == before
        issues = [f.get('issue') for f in result['findings']]
        assert 'empty scopes' in issues and 'overlapping active scopes' in issues
        assert any('error' in f for f in result['findings'])
    print('self-test passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(inspect(args.root), ensure_ascii=False, indent=2))
