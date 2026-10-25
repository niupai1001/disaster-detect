"""Dense F tiles need a geographic guard, beyond merely disjoint row IDs."""
import importlib.util
from pathlib import Path

from shapely.geometry import box

spec=importlib.util.spec_from_file_location('prepare_f',Path(__file__).resolve().parents[1]/'scripts/prepare_use_f_local.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_geographic_split_separates_neighbours_and_is_stable():
    rows=[]
    for y in range(32):
        for x in range(32):
            rows.append(dict(file=f'{y}_{x}',bbox_projected=[x*1280,y*1280,(x+1)*1280,(y+1)*1280],group=f'{x//4}:{y//4}'))
    train,validation,guard=module.geographic_split(rows)
    sets=[{r['file'] for r in group} for group in (train,validation,guard)]
    assert set.union(*sets)=={r['file'] for r in rows}
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert train and validation and guard
    for sample in train:
        assert not any(box(*sample['bbox_projected']).intersects(box(*v['bbox_projected']).buffer(1280)) for v in validation)
    repeated=module.geographic_split(list(reversed(rows)))
    assert [{r['file'] for r in group} for group in repeated]==sets
