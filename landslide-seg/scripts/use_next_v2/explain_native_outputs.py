"""只读：一张验证块进入冻结 DOFA / AnySat 后各层实际输出（形状、统计、可视化）。

不训练、不改权重、不写预测目录；只按 readout.py 的实际前向复现一次，
用于回答“我的图像进 DOFA/AnySat 后到底得到什么”。
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import yaml
from torch.nn import functional as F

for _font in ('C:/Windows/Fonts/NotoSansSC-VF.ttf', 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf'):
    if Path(_font).is_file():
        font_manager.fontManager.addfont(_font)
        matplotlib.rcParams['font.family'] = font_manager.FontProperties(fname=_font).get_name()
        break
matplotlib.rcParams['axes.unicode_minus'] = False

torch.set_num_threads(8)
from landslide_seg.data.segformer_dataset import load_stats, read_manifest, collate
from landslide_seg.data.use_dataset import USESegDataset
from landslide_seg.models.use_next_v2 import EOAsset
from landslide_seg.models.use_next_v2.anchors import S2_BANDS
from landslide_seg.models.use_next_v2.readout import NativeReadout, state_digest

OUT = ROOT / 'outputs/use_next_v2/explain_native'
KINDS = ('dofa', 'anysat')


def tile_asset(model, batch, index):
    """与 NativeReadout.forward 完全相同的单样本输入构造。"""
    x = batch['x'][index]
    spatial = batch['spatial_metadata'][index]
    mean = x.new_tensor(model.stats['mean'])[:, None, None]
    std = x.new_tensor(model.stats['std'])[:, None, None]
    physical = (x * std + mean) * .0001
    valid = batch['pixel_valid'][index]
    return EOAsset('s2', S2_BANDS, torch.where(valid, physical, 0), batch['band_metadata'][index],
                   batch['known_mask'][index], {k: spatial[k] for k in ('crs', 'transform')},
                   None, None, valid, spatial['date'], 'surface_reflectance', spatial)


def stats_of(tensor):
    t = tensor.detach().float()
    return dict(shape=list(t.shape), dtype=str(tensor.dtype), min=float(t.min()), max=float(t.max()),
                mean=float(t.mean()), std=float(t.std()),
                finite=bool(torch.isfinite(t).all()))


def pca_rgb(maps, size):
    """maps: (C,h,w) → size×size 三主成分伪彩；逐通道标准化后取分量2–98百分位。"""
    c, h, w = maps.shape
    m = maps.detach().float().reshape(c, -1)
    m = (m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-6)
    u, s, _ = torch.linalg.svd(m.T, full_matrices=False)   # 像素×通道，按像素打分取主成分
    scores = (u[:, :3] * s[:3]).T.reshape(3, h, w)
    scores = F.interpolate(scores[None], size=(size, size), mode='bilinear', align_corners=False)[0]
    out = []
    for comp in scores:
        lo, hi = torch.quantile(comp.flatten(), torch.tensor([.02, .98]))
        out.append(((comp - lo) / (hi - lo + 1e-6)).clamp(0, 1))
    return torch.stack(out).permute(1, 2, 0).numpy()


def rgb_composite(physical, valid):
    """第2/3/8 波段（红/绿/蓝）真彩色，2–98 百分位拉伸。"""
    pick = [S2_BANDS.index(b) for b in ('red', 'green', 'blue')]
    img = physical[pick].clone()
    for c in range(3):
        v = img[c][valid[0]]
        lo, hi = torch.quantile(v, torch.tensor([.02, .98])) if v.numel() else (0., 1.)
        img[c] = ((img[c] - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    facts = {'purpose': '只读单块前向：冻结原生 DOFA / AnySat 的实际输出形态', 'formal_updates': 0}
    figure = {}
    for kind in KINDS:
        run = ROOT / f'outputs/use_next_v2/runs/p1_{kind}'
        cfg = yaml.safe_load((run / 'resolved_config.yaml').read_text(encoding='utf-8'))
        data = cfg['segformer']['data']
        stats = load_stats(ROOT / data['stats'])
        rows = read_manifest(ROOT / data['manifest'])
        val = USESegDataset([r for r in rows if r['split'] == 'val'], stats, augment=False, preload=False)
        sample = val[0]
        batch = collate([sample])
        model = NativeReadout(ROOT, kind, stats)
        saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        asset = tile_asset(model, batch['image'], 0)
        physical, valid = asset.image, asset.pixel_valid
        wavelengths = asset.channel_metadata[:, 0].tolist()

        with torch.no_grad():
            began = time.perf_counter()
            raw = model.anchor(asset)
            record_latency = time.perf_counter() - began
            if kind == 'dofa':
                levels = list(raw)
            else:
                dense = raw.permute(0, 3, 1, 2)
                levels = list(dense.split(768, dim=1))
            projected = [F.interpolate(p(l), size=physical.shape[-2:], mode='bilinear', align_corners=False)
                         for p, l in zip(model.projections, levels)]
            stack = torch.stack(projected).mean(0)
            logits = model.head(stack)
            prob = logits[0, 0].sigmoid()

        record = {
            'tile': dict(sample_id=sample['sample_id'], event_id=sample['event_id'],
                         date=asset.acquisition_time, bbox=sample.get('bbox'),
                         input_shape=list(physical.shape), wavelengths_um=wavelengths),
            'anchor': dict(kind=kind, digest=state_digest(model.anchor), frozen=True,
                           forward_seconds_cpu=record_latency),
            'levels': [stats_of(t) for t in levels],
            'projected': [stats_of(t) for t in projected],
            'stacked': stats_of(stack), 'logits': stats_of(logits), 'prob': stats_of(prob),
            'readout_params': dict(total=sum(p.numel() for p in model.parameters()),
                                   trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
        }
        if kind == 'dofa':
            record['levels_note'] = ('官方 vit_base_patch16(img_size=224) out_indices=[3,5,7,11]：14×14 网格；'
                                     '每格16像素，在1280米块上约91米；输入由128×128双线性插值到224×224')
        else:
            ctx, loc = levels
            blocks = ctx.reshape(768, 32, 4, 32, 4)
            record['anysat_split'] = {
                'dense_shape': list(raw.shape),
                'patch_size_m': 40, 'patch_grid': [32, 32], 'subpatch_grid': [4, 4],
                'context_within_patch_std_max': float(blocks.std(-1).std(-2).max()),
                'local_within_patch_std_mean': float(loc.reshape(768, 32, 4, 32, 4).std(-1).std(-2).mean()),
                'note': '前768通道为Transformer上下文（每个40米patch一个向量），后768为投影器局部子patch向量（10米）',
            }

        reference = ROOT / f'outputs/use_next_v2/audit/p1_predictions/{kind}/best/0000.npz'
        if reference.is_file():
            with np.load(reference) as npz:
                stored = npz['prob']
                record['stored_prediction'] = dict(path=str(reference.relative_to(ROOT)),
                                                   max_abs_diff=float(np.abs(stored - prob.numpy()).max()))
        gt = sample['mask'].numpy().astype(bool)
        v = sample['valid'].numpy().astype(bool)
        pred = prob.numpy() >= .5
        inter = float((pred & gt & v).sum())
        record['tile_iou_at_0.5'] = inter / float(((pred | gt) & v).sum() + 1e-9)
        record['gt_positive_pixels'] = int((gt & v).sum())
        record['valid_fraction'] = float(v.mean())
        record['probability_inside_valid'] = dict(mean=float(prob.numpy()[v].mean()),
                                                 above_half=float((pred[v]).mean()))
        record['probability_outside_valid'] = dict(mean=float(prob.numpy()[~v].mean()) if (~v).any() else None)

        if kind == 'dofa':
            composite = rgb_composite(physical, v[None])
            composite[~v] = 0.45
            figure['input'] = composite
            figure['mask'] = np.where(v, gt.astype(float), np.nan)
            figure['dofa_pca'] = pca_rgb(levels[-1][0], 128)
        else:
            figure['anysat_context'] = pca_rgb(levels[0][0], 128)
            figure['anysat_local'] = pca_rgb(levels[1][0], 128)
        figure[f'{kind}_stack'] = pca_rgb(stack[0], 128)
        figure[f'{kind}_prob'] = np.where(v, prob.numpy(), np.nan)
        facts[kind] = record
        print(json.dumps({k: v for k, v in record.items() if k in ('tile', 'levels', 'prob', 'stored_prediction')},
                         ensure_ascii=False)[:1200], flush=True)

    names = [('input', '输入：S2十波段真彩色（灰＝无效像元）'),
             ('mask', '标注：滑坡真值（灰＝无效像元）'),
             ('dofa_pca', 'DOFA 第4级原生特征（768维→3主成分）'),
             ('dofa_stack', 'DOFA 读入32通道融合特征（4级投影均值）'),
             ('anysat_context', 'AnySat 上下文段（每40米patch恒定）'),
             ('anysat_local', 'AnySat 局部段（10米子patch特征）'),
             ('dofa_prob', 'DOFA 滑坡概率（仅有效区，灰＝无效）'),
             ('anysat_prob', 'AnySat 滑坡概率（仅有效区，灰＝无效）')]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    for ax, (key, title) in zip(axes.ravel(), names):
        image = figure[key]
        ax.set_title(title, fontsize=11)
        if image.ndim == 3:
            ax.imshow(image)
        elif key.endswith('prob'):
            palette = plt.get_cmap('inferno').copy()
            palette.set_bad('0.72')
            ax.imshow(np.ma.masked_invalid(image), cmap=palette, vmin=0, vmax=1)
        else:
            palette = plt.get_cmap('gray').copy()
            palette.set_bad('0.72')
            ax.imshow(np.ma.masked_invalid(image), cmap=palette, vmin=0, vmax=1)
        ax.axis('off')
    fig.suptitle('一张1280米验证块进入冻结 DOFA / AnySat 后的实际输出'
                 '（伪彩为特征主成分，不是新模态影像；灰色区域不计分）',
                 fontsize=13)
    fig.tight_layout()
    figure_path = OUT / 'native_outputs.png'
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    facts['figure'] = str(figure_path.relative_to(ROOT))
    facts['caveat'] = ('本项目只加载官方编码器权重并读取密集特征；没有解码器/重建头，'
                       '因此输出不是“另一模态的影像”，而是同一模态的特征图与单通道滑坡概率。')
    (OUT / 'outputs.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[DONE]', figure_path)


if __name__ == '__main__':
    main()
