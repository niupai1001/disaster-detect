"""Portable TVLP 2.0 writer. Copy this file into any Python training project."""
import json
import math
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


class RunWriter:
    def __init__(self, directory, *, experiment=None, variant=None, metrics=None,
                 charts=None, numbers=None, primary_metric=None, config=None, resume=False):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path / '.writer.lock'
        self.guard = threading.RLock()
        self.stop = threading.Event()
        self.closed = False
        self.seq = 0
        self.step = -1
        self.session = uuid.uuid4().hex
        # OS lock is released even when a training process crashes.
        self.lock_file = open(self.lock_path, 'a+b')
        self.lock_file.seek(0)
        if self.lock_file.seek(0, 2) == 0:
            self.lock_file.write(b'0')
            self.lock_file.flush()
        self.lock_file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta_path = self.path / 'run.json'
            if resume:
                self.meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if self.meta['schema_version'] != '2.0':
                    raise ValueError('Only TVLP 2.0 can be resumed')
                for name in ('metrics.jsonl', 'events.jsonl'):
                    p = self.path / name
                    if p.exists():
                        data = p.read_bytes()
                        # Remove only the incomplete trailing record before resuming.
                        if data and not data.endswith(b'\n'):
                            data = data[:data.rfind(b'\n') + 1]
                            p.write_bytes(data)
                        for line in data.splitlines():
                            try:
                                self.seq = max(self.seq, int(json.loads(line)['seq']))
                            except (ValueError, KeyError, TypeError):
                                continue
            else:
                if meta_path.exists():
                    raise ValueError('Run already exists; use resume=True or a new directory')
                if not experiment or not variant or not metrics:
                    raise ValueError('experiment, variant and metrics are required')
                if variant['kind'] != 'baseline' and not variant.get('description', '').strip():
                    raise ValueError('A changed variant requires a description')
                charts = charts or list(metrics)[:2]
                numbers = numbers or []
                primary_metric = primary_metric or charts[0]
                if not 1 <= len(charts) <= 4 or len(numbers) > 4:
                    raise ValueError('Choose 1–4 charts and up to 4 numbers')
                if any(k not in metrics for k in charts + numbers + [primary_metric]):
                    raise ValueError('View references an undefined metric')
                for definition in metrics.values():
                    if definition['direction'] not in ('min', 'max', 'none') or definition['role'] not in ('loss', 'score', 'other'):
                        raise ValueError('Invalid metric direction or role')
                self.meta = dict(schema_version='2.0', run_id=uuid.uuid4().hex,
                                 created_at=now(), experiment=experiment, variant=variant,
                                 metrics=metrics, config=config or {},
                                 view=dict(charts=charts, numbers=numbers, primary_metric=primary_metric))
                temporary = meta_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(self.meta, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                temporary.replace(meta_path)
            self._event('resumed' if resume else 'started')
            self.thread = threading.Thread(target=self._heartbeat, daemon=True)
            self.thread.start()
        except Exception:
            self.lock_file.close()
            raise

    def _append(self, filename, payload):
        if self.closed:
            raise RuntimeError('Writer is closed')
        self.seq += 1
        payload.update(seq=self.seq, session_id=self.session, time=now())
        with open(self.path / filename, 'a', encoding='utf-8') as file:
            file.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
            file.flush()

    def _event(self, kind):
        with self.guard:
            self._append('events.jsonl', dict(kind=kind))

    def _heartbeat(self):
        while not self.stop.wait(10):
            try:
                self._event('heartbeat')
            except (OSError, RuntimeError):
                return

    def log(self, step, values, epoch=None):
        with self.guard:
            if isinstance(step, bool) or not isinstance(step, int) or step < max(0, self.step):
                raise ValueError('step must be a nonnegative, nondecreasing integer')
            if not values or any(k not in self.meta['metrics'] or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for k, v in values.items()):
                raise ValueError('Values must be declared finite numeric metrics')
            record = dict(step=step, values=values)
            if epoch is not None:
                if isinstance(epoch, bool) or not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch < 0:
                    raise ValueError('epoch must be finite and nonnegative')
                record['epoch'] = epoch
            self._append('metrics.jsonl', record)
            self.step = step

    def finish(self, failed=False):
        self.stop.set()
        with self.guard:
            if not self.closed:
                try:
                    self._event('failed' if failed else 'completed')
                finally:
                    self.closed = True
                    self.lock_file.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.finish(failed=kind is not None)
