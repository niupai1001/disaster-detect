"""Build observation-only quality, without opening source label arrays."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BANDS = ('B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12')


def main():
    output = ROOT / 'outputs/use/quality'
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (ROOT/'manifests/v4r1/pool_100.jsonl').read_text().splitlines()]
    index = {}
    for i, rec in enumerate(records):
        source = Path(rec['source_uri'])
        with h5py.File(source, 'r') as f:
            t = rec['time_index']
            x = np.stack([f[b][t] for b in BANDS])
            if 'SCL' not in f:
                raise ValueError(f'Missing source quality: {source}')
            quality = np.isfinite(x) & (x > 0) & np.isin(f['SCL'][t], (4,5,6,7))[None]
        cache = Path(rec['cache'])
        if hashlib.sha256(cache.read_bytes()).hexdigest() != rec['cache_sha256']:
            raise ValueError('Frozen cache hash changed: '+rec['sample_id'])
        with np.load(cache) as cached:
            if not np.array_equal(cached['x10'], x.astype(np.int16)):
                raise ValueError('Source observation differs from cached input: '+rec['sample_id'])
        name = rec['sample_id'] + '.npz'
        target = output / name
        if target.exists():
            with np.load(target) as old:
                if not np.array_equal(old['pixel_valid'], quality):
                    raise ValueError('Existing sidecar differs; do not overwrite')
        else:
            np.savez_compressed(target, pixel_valid=quality)
        index[rec['sample_id']] = dict(file=name, sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            cache_sha256=rec['cache_sha256'], time_index=t, source_uri=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if i % 100 == 0:
            print(f'{i}/{len(records)}', flush=True)
    temp = output/'index.tmp'
    temp.write_text(json.dumps(index, indent=1), encoding='utf-8')
    temp.replace(output/'index.json')


if __name__ == '__main__':
    main()
