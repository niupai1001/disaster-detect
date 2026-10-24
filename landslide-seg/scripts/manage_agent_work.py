"""Small shared work register; claims never start experiments or change research facts."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REGISTER = ROOT / 'docs/assets/research/agent_work.json'


def atomic_replace(source, target):
    # Windows readers/antivirus can briefly hold a JSON file without delete-sharing.
    for attempt in range(50):
        try:
            os.replace(source,target)
            return
        except PermissionError:
            if attempt==49:raise
            time.sleep(.1)


@contextmanager
def locked(path):
    lock = path.with_suffix('.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(f'登记正在写入或上次写入中断；核验持有者后再处理锁：{lock}')
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink()


def update(path, action, task_id, owner, title=None, scopes=None, evidence=None, note=None):
    if not owner.strip() or not task_id.strip():
        raise ValueError('任务和负责人不能为空')
    with locked(path):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        tasks = data['tasks']
        task = next((t for t in tasks if t['id'] == task_id), None)
        now = datetime.now(timezone.utc).isoformat()
        if action == 'claim':
            if task and not (task['status'] == 'planned' or
                             task['status'] == 'blocked' and task['owner'] == owner):
                raise ValueError('任务已登记；先核对已有成果/持有者，不能重复认领或自动接管')
            if not title or not scopes or any(not s.strip() for s in scopes):
                raise ValueError('认领需要 --title 和至少一个非空 --scope（文件路径或实验输出目录）')
            scopes = sorted({s.replace('\\', '/').rstrip('/').casefold() for s in scopes})
            if any(not s or ':' in s or PurePosixPath(s).is_absolute() or
                   '..' in PurePosixPath(s).parts or s == '.' for s in scopes):
                raise ValueError('范围必须是相对项目根目录的路径，不能包含盘符、.. 或根目录本身')
            scopes = sorted({str(PurePosixPath(s)) for s in scopes})
            for other in tasks:
                if other['status'] != 'active':
                    continue
                for left in scopes:
                    for right in other['scopes']:
                        right = right.replace('\\', '/').rstrip('/').casefold()
                        if left == right or left.startswith(right + '/') or right.startswith(left + '/'):
                            raise ValueError(f"范围已被 {other['owner']} 的 {other['id']} 占用：{left}")
            if task is None:
                task = {'id': task_id}
                tasks.append(task)
            task.update(title=title, owner=owner, scopes=scopes, status='active', started_at=now)
        else:
            if task is None or task['status'] != 'active' or task['owner'] != owner:
                raise ValueError('只能由登记持有者结束正在进行的任务')
            if not evidence or not note:
                raise ValueError('结束任务必须给 --evidence 和 --note；受阻要写原因与下一步')
            task.update(status={'finish':'completed','block':'blocked','note':'active'}[action], evidence=evidence, note=note)
        task['updated_at'] = now
        data['updated_at'] = now
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        atomic_replace(temp, path)
        return task


def transfer(path, task_id, owner, source_id, scopes, note):
    """Explicit human-authorized scope handoff, preserving the source task."""
    if not scopes or not note or not source_id or not owner:
        raise ValueError('移交需要来源任务、范围与用户授权说明')
    scopes = {s.replace('\\', '/').rstrip('/').casefold() for s in scopes}
    with locked(path):
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        source = next(t for t in data['tasks'] if t['id'] == source_id)
        target = next(t for t in data['tasks'] if t['id'] == task_id)
        if source is target or source['status'] != 'active' or target['status'] != 'active' or target['owner'] != owner:
            raise ValueError('移交双方必须活动，且接收方身份一致')
        if not scopes.issubset(set(source['scopes'])):
            raise ValueError('只能移交来源任务完整登记的范围')
        for other in data['tasks']:
            if other is source or other is target or other['status'] != 'active':
                continue
            if any(a == b or a.startswith(b+'/') or b.startswith(a+'/') for a in scopes for b in other['scopes']):
                raise ValueError('移交范围还与第三方冲突')
        now = datetime.now(timezone.utc).isoformat()
        source['scopes'] = sorted(set(source['scopes']) - scopes)
        target['scopes'] = sorted(set(target['scopes']) | scopes)
        record = dict(at=now, source=source_id, target=task_id, scopes=sorted(scopes), authorization=note)
        data.setdefault('transfers', []).append(record)
        source['updated_at'] = target['updated_at'] = data['updated_at'] = now
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        atomic_replace(temp, path)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'claim', 'finish', 'block', 'transfer', 'note'])
    parser.add_argument('--from-task')
    parser.add_argument('task_id', nargs='?')
    parser.add_argument('--owner', help='聊天ID或可追溯的Agent身份')
    parser.add_argument('--title')
    parser.add_argument('--scope', action='append', help='相对项目根目录路径；重复此参数登记多个范围')
    parser.add_argument('--evidence', action='append')
    parser.add_argument('--note')
    parser.add_argument('--limit', type=int, default=10, help='列表最多显示的已结束任务；活动任务始终显示')
    args = parser.parse_args()
    if args.action == 'list':
        data = json.loads(REGISTER.read_text(encoding='utf-8-sig'))
        rows = data['tasks']
        if args.task_id:
            rows = [t for t in rows if args.task_id in t['id'] or args.task_id in t['title']]
        active = [t for t in rows if t['status'] in ('active', 'blocked')]
        ended = [t for t in rows if t['status'] not in ('active', 'blocked')]
        print(json.dumps({'sources': data['sources'], 'matches': len(rows),
                          'tasks': active + ended[-max(1, args.limit):]}, ensure_ascii=False, indent=2))
        return
    if not args.task_id or not args.owner:
        parser.error('写入必须提供 task_id 与 --owner')
    try:
        if args.action == 'transfer':
            task = transfer(REGISTER, args.task_id, args.owner, args.from_task, args.scope, args.note)
        else:
            task = update(REGISTER, args.action, args.task_id, args.owner, args.title,
                          args.scope, args.evidence, args.note)
    except ValueError as exc:
        parser.exit(1, str(exc) + '\n')
    print(json.dumps(task, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
