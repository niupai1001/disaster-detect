"""Package completed v4r3 sources/provenance/results without datasets or model weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    audit = json.loads((ROOT / 'reports/v4r3_results/completion_audit.json').read_text(encoding='utf-8'))
    assert audit['status'] == 'PASS' and audit['n_new_formal_runs'] == 2
    dofa = next(r for r in audit['runs'] if r['condition'] == 'dofa_upernet_pyramid')
    for relative, digest in dofa['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    members = set()
    for relative in ('src', 'configs', 'tests', 'manifests/v4r1', 'docs', 'scripts',
                     'vendor/dofa-official-0cfb7e1', 'vendor/dofa_runtime'):
        for path in (ROOT / relative).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pyd', '.dll'):
                members.add(path)
    for path in (ROOT / 'reports').glob('v4r3*'):
        if path.name == 'v4r3_archive_identity.json':
            continue
        members.update(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else members.add(path)
    for run in audit['runs']:
        directory = Path(run['run_dir'])
        for name in ('train_summary.json', 'run_meta.json', 'resolved_config.json', 'sample_exposures.json'):
            members.add(directory / name)
        members.update(path for path in (directory / 'eval').iterdir() if path.suffix in ('.json', '.csv'))
    for pattern in ('v4r3*.log',):
        members.update((ROOT / 'outputs/segformer').glob(pattern))
    for relative in ('outputs/segformer/v4r3_low_encoder_lr_batch', 'outputs/segformer/v4r3_dofa_batch'):
        members.update(path for path in (ROOT / relative).iterdir() if path.is_file())
    for relative in ('vendor/dofa-official-0cfb7e1.source.zip', 'vendor/dofa_wavelength_mapping_terratorch_1_2_13.json',
                     'weights/dofa/XShadow_DOFA_base/weight_identity.json', 'manifests/split_v4.json'):
        members.add(ROOT / relative)
    members.add(ROOT / 'reports/v4r1_results/completion_audit.json')
    package = ROOT / 'outputs/segformer/v4r3_final_sources_and_results.zip'
    identities = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(members)}
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(members):
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr('v4r3_package_manifest.json', json.dumps(identities, indent=1))
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
    report = dict(path=str(package), sha256=sha(package), size_bytes=package.stat().st_size,
                  n_files=len(members), includes_datasets=False, includes_model_weights=False,
                  preserves_both_run_source_snapshots=True, current_dofa_source_matches_formal_training=True)
    (ROOT / 'reports/v4r3_archive_identity.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
