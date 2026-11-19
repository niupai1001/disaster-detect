"""Index PD-HERE run evidence and expose completed weights by architecture."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'configs/pd_here_s60/versions.json'
TRAINING_POOLS = {
    700: ('fixed700', '700景固定窗口'),
    2000: ('subset2000', '2000子集'),
    13020: ('full13020', '全量训练'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def confirmed_version(model_id):
    return next(version for version in read(CATALOG)['versions']
                if version['name_confirmed'] and model_id in version['model_ids'])


def viewer_labels(version, parents):
    pool, label = TRAINING_POOLS[parents]
    experiment = dict(id=f'pd_here_s60_{pool}', name=label)
    variant = dict(id=f'{pool}-{version["key"]}', name=f'{label}｜{version["name"]}')
    return experiment, variant


def sync_viewer_labels(path, version, parents):
    path = path / 'run.json'
    if not path.is_file():
        return
    meta = read(path)
    if meta['config']['model_id'] not in version['model_ids']:
        raise ValueError(f'{path}: Viewer model differs from its architecture registry')
    experiment, variant = viewer_labels(version, parents)
    updated = dict(meta, experiment=meta['experiment'] | experiment, variant=meta['variant'] | variant)
    if updated == meta:
        return
    with NamedTemporaryFile('w', dir=path.parent, suffix='.tmp', delete=False, encoding='utf-8') as file:
        temporary = Path(file.name)
        json.dump(updated, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def index_versions():
    catalog = read(CATALOG)
    logs = ROOT / catalog['logs_root']
    weights = ROOT / catalog['weights_root']
    viewer = ROOT / catalog['viewer_root']
    known_models = {model_id for version in catalog['versions'] for model_id in version['model_ids']}
    runs_by_model = {}
    for path in logs.glob('*/identity.json'):
        identity = read(path)
        if identity.get('model_id') in known_models:
            runs_by_model.setdefault(identity['model_id'], []).append(path.parent.name)
    registered = set()
    for version in catalog['versions']:
        folder = weights / version['key']
        folder.mkdir(parents=True, exist_ok=True)
        fixed = version.get('fixed_window', {})
        records = []
        new_runs = sorted({run for model_id in version['model_ids']
                           for run in runs_by_model.get(model_id, [])} - set(version['runs']))
        for run in [*version['runs'], *new_runs]:
            if run in registered:
                raise ValueError(f'Run appears under more than one architecture: {run}')
            registered.add(run)
            directory = logs / run
            status = read(directory / 'status.json')
            identity = read(directory / 'identity.json') if (directory / 'identity.json').is_file() else {}
            model_id = identity.get('model_id')
            if model_id is not None and model_id not in version['model_ids']:
                raise ValueError(f'{run}: {model_id} differs from its architecture registry')
            is_fixed = model_id == fixed.get('model_id') if fixed else False
            implementation = fixed['implementation'] if is_fixed else version['implementation']
            parents = fixed['train_parents'] if is_fixed else identity.get('parents')
            display_name = None
            if parents in TRAINING_POOLS:
                _, display = viewer_labels(version, parents)
                display_name = display['name']
                sync_viewer_labels(viewer/run, version, parents)
            checkpoints = {}
            own_weights = folder / run
            separated = 'weights' in identity
            if separated and identity['weights'] != relative(own_weights):
                raise ValueError(f'{run}: checkpoint directory differs from its architecture registry')
            for source in sorted((own_weights if separated else directory).glob('*.pt')):
                item = {'original': relative(source)}
                # New last/best files are atomically replaced while a run is in progress.
                if separated:
                    item.update(weight=relative(source), bytes=source.stat().st_size)
                elif status['status'] == 'COMPLETED':
                    destination = own_weights / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        os.link(source, destination)
                    if not destination.samefile(source):
                        raise ValueError(f'Existing weight points to another run: {destination}')
                    item.update(weight=relative(destination), bytes=source.stat().st_size)
                checkpoints[source.name] = item
            records.append(dict(
                run=run, model_id=model_id, state=status['status'], implementation=implementation,
                parents=parents, epochs=status.get('epochs'), viewer_name=display_name,
                logs=relative(directory), viewer=relative(viewer/run) if (viewer/run).is_dir() else None,
                checkpoints=checkpoints,
            ))
        implementation = ROOT / version['implementation']
        payload = dict(version, implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest(),
                       runs=records)
        target = folder / 'run_index.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        temporary.replace(target)
        pinned = sum('weight' in item for record in records for item in record['checkpoints'].values())
        print(f"{version['key']}: {len(records)} runs, {pinned} completed checkpoints")


if __name__ == '__main__':
    index_versions()
