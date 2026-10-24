"""只读审计用户指定外部影像/标注；不推断波段身份、不推理或改标签。"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parents[1]
IMAGES = Path('N:/Users/yzh/Data/滑坡样本/滑坡样本')
MASKS = Path('N:/Users/yzh/Data/滑坡mask/滑坡mask/select_train_masks')


def main():
    imgs = {p.stem: p for p in IMAGES.glob('*.tif')}
    masks = {p.stem.removesuffix('_mask'): p for p in MASKS.glob('*.tif')}
    pairs = sorted(imgs.keys() & masks.keys())
    records, duplicates = [], defaultdict(list)
    for sid in pairs:
        ip, mp = imgs[sid], masks[sid]
        with rasterio.open(ip) as im, rasterio.open(mp) as mask:
            x, y = im.read(), mask.read()
            same = im.shape == mask.shape and im.crs == mask.crs and im.transform == mask.transform
            digest = hashlib.sha256(ip.read_bytes()).hexdigest()
            duplicates[digest].append(sid)
            row = {'id': sid, 'image': str(ip), 'mask': str(mp), 'image_sha256': digest,
                   'mask_sha256': hashlib.sha256(mp.read_bytes()).hexdigest(),
                   'shape': list(im.shape), 'bands': im.count, 'descriptions': list(im.descriptions),
                   'crs': str(im.crs), 'transform': list(im.transform), 'aligned': same,
                   'bounds_wgs84': list(transform_bounds(im.crs, 'EPSG:4326', *im.bounds)),
                   'image_nodata': im.nodata, 'nonfinite_values': int((~np.isfinite(x)).sum()),
                   'invalid_pixels': int((im.dataset_mask() == 0).sum()),
                   'mask_values': [int(v) for v in np.unique(y)],
                   'positive_pixels': int((y == 1).sum()), 'total_pixels': int(y.size),
                   'band_min': [float(np.nanmin(b)) for b in x],
                   'band_max': [float(np.nanmax(b)) for b in x], 'tags': im.tags()}
            records.append(row)
    boxes = [r['bounds_wgs84'] for r in records]
    overlap_pairs = []
    for i, a in enumerate(boxes):
        for j in range(i):
            b = boxes[j]
            if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                overlap_pairs.append([records[j]['id'], records[i]['id']])
    existing = []
    for line in (ROOT / 'manifests/v4r1/pool_100.jsonl').read_text(encoding='utf-8-sig').splitlines():
        r = json.loads(line)
        existing.append((r['sample_id'], transform_bounds(r['crs'], 'EPSG:4326', *r['bbox'])))
    parent_overlap = []
    for r, a in zip(records, boxes):
        hits = [sid for sid, b in existing if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])]
        if hits:
            parent_overlap.append({'id': r['id'], 'existing_ids': hits})
    out = {'status': 'REFERENCE_ONLY_EXCLUDED_FROM_MAINLINE',
           'user_clarification': {
               'bands_in_order': ['B', 'G', 'R', 'NIR', 'evaluation', 'slope', 'aspect', 'cerve', 'TPI'],
               'source': '用户提供顺序；evaluation/cerve保留原写法，单位与算法未确认',
               's2_correspondence': False, 'dates_available': False,
               'annotation_completeness': '用户认为应该标注完，非独立复核确认',
               'role': '仅可能参考数据，不纳入主线', 'inference_performed': False},
           'image_count': len(imgs), 'mask_count': len(masks), 'paired_count': len(pairs),
           'images_without_mask': sorted(imgs.keys()-masks.keys()),
           'masks_without_image': sorted(masks.keys()-imgs.keys()),
           'band_counts': dict(Counter(r['bands'] for r in records)),
           'crs_counts': dict(Counter(r['crs'] for r in records)),
           'aligned_pairs': sum(r['aligned'] for r in records),
           'nonfinite_values': sum(r['nonfinite_values'] for r in records),
           'invalid_pixels': sum(r['invalid_pixels'] for r in records),
           'mask_values': sorted({v for r in records for v in r['mask_values']}),
           'positive_pairs': sum(r['positive_pixels'] > 0 for r in records),
           'pixel_foreground_fraction': sum(r['positive_pixels'] for r in records)/sum(r['total_pixels'] for r in records),
           'duplicate_image_hashes': [v for v in duplicates.values() if len(v)>1],
           'internal_bbox_overlap_pairs': overlap_pairs, 'parent_pool_bbox_overlap': parent_overlap,
           'overlap_scope': '仅与原TRAIN/DEV清单2892块做地理包围盒比较；不证明事件/来源独立，不含其他历史或预训练语料',
           'boundaries': ['用户确认通道不对应S2且无日期；传感器、单位/偏移未确认，不能接现有十波段模型',
                          '无nodata声明和非有限值不代表无云/影/无效覆盖',
                          '标注完整性待确认，零像素不自动当可靠负例',
                          '原始文件不改动；未运行模型，不用新集调参'],
           'records': records}
    (ROOT/'docs/assets/research/external_landslide_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k not in ('records','internal_bbox_overlap_pairs')},ensure_ascii=False))
    print('internal overlapping bbox pairs',len(overlap_pairs))
    assert len(records) == len(pairs)


if __name__ == '__main__':
    main()
