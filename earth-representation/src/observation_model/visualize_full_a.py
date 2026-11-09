"""Read-only comparisons on fixed held-out A targets; use the executed source snapshot."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'observation_model/outputs/token_A_full'
sys.path.insert(0, str(OUT / 'source_v2/earth-representation'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from observation_model.full_image_data import read_full
from observation_model.train_full_image_a import inputs
from observation_model.token_prototype import PatchModel
from observation_model.full_a_controls import features


def main():
    torch.set_num_threads(4)
    torch.cuda.init()
    destination = OUT / 'prediction_visuals'
    destination.mkdir(exist_ok=True)
    state = torch.load(OUT / 'best.pt', map_location='cuda', weights_only=False)
    fits = torch.load(OUT / 'spectral_fits.pt', map_location='cuda', weights_only=False)
    stats = state['statistics']
    model = PatchModel().cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    members = json.loads((OUT / 'members.json').read_text())
    result = json.loads((OUT / 'result.json').read_text())
    controls = json.loads((OUT / 'controls.json').read_text())
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    evidence = []
    for sensor, bands in [('S2', ['S2:B02', 'S2:B8A', 'S2:B11']),
                          ('L8', ['L8:SR_B2', 'L8:SR_B5', 'L8:SR_B6'])]:
        chosen = [next(r for r in members if r['split'] == 'val' and r['target_band'] == b) for b in bands]
        fig, axes = plt.subplots(3, 8, figsize=(22, 9), layout='constrained')
        for row, r in enumerate(chosen):
            arrays, actual = read_full(r)
            assert all(a['input_sha256'] == b['input_sha256'] and a['valid_sha256'] == b['valid_sha256']
                       for a, b in zip(actual['full_bands'], r['full_bands']))
            items, hidden = inputs(dict(arrays=arrays, record=r), stats)
            j = r['target_index']; b = items[j]; y, x = r['mask_patch_yx']; p = 60 // b.native_gsd_m
            ys, xs = slice(y*p, (y+1)*p), slice(x*p, (x+1)*p)
            with torch.no_grad():
                prediction = model(items, hidden)[j][0, ys, xs]
                yy, xx = torch.meshgrid(torch.arange(b.value.shape[-2], device='cuda'), torch.arange(b.value.shape[-1], device='cuda'), indexing='ij')
                xy = torch.stack([yy, xx], -1).float(); keep = b.valid[0].clone(); keep[ys, xs] = False
                distances, idx = torch.cdist(xy[ys, xs].reshape(-1, 2), xy[keep]).topk(4, largest=False)
                weights = distances.clamp_min(1e-6).pow(-2)
                spatial = ((b.value[0][keep][idx]*weights).sum(1)/weights.sum(1)).reshape(p, p)
                a, valid_features = features(items, r); f = fits[b.name]
                a = torch.where(valid_features, a, f['mean'].float())
                spectral = ((a.double()-f['mean']) @ f['w'] + f['ym']).float()
            valid = b.valid[0, ys, xs]; truth = b.value[0, ys, xs]
            normalized = {'spatial': spatial, 'spectral': spectral, 'transformer': prediction}
            mse = {key: float(((value-truth)[valid]).square().mean()) for key, value in normalized.items()}
            for key, value in mse.items():
                rows = result['best_model']['val_rows'] if key == 'transformer' else controls['val'][key]['rows']
                saved = next(t['mse'] for t in rows if t['id'] == r['observation_id'])
                assert abs(value-saved) < 1e-5, (key, value, saved)
            s = stats[b.name]
            physical = {key: (value*s['std']+s['mean']).cpu().numpy() for key, value in normalized.items()}
            target = (truth*s['std']+s['mean']).cpu().numpy(); q = valid.cpu().numpy()
            context = (b.value[0]*s['std']+s['mean']).cpu().numpy()
            context = np.where(b.valid[0].cpu().numpy(), context, np.nan)
            pixels = np.concatenate([target[q]] + [v[q] for v in physical.values()])
            lo, hi = float(pixels.min()), float(pixels.max())
            if hi-lo < 1e-5: hi = lo+1e-5
            ax = axes[row, 0]
            ax.imshow(context, cmap='gray', vmin=np.nanpercentile(context, 2), vmax=np.nanpercentile(context, 98), interpolation='nearest')
            ax.add_patch(Rectangle((x*p-.5, y*p-.5), p, p, edgecolor='red', facecolor='none', linewidth=1.5))
            ax.set_title(f"{b.name} 原图\n{r['date'][:10]}")
            ax.set_ylabel(f'{b.native_gsd_m}m像元；缺口{p}×{p}\n验证案例 {r["observation_id"][-24:]}', fontsize=8)
            images = [('真实目标', target)] + list(zip(['空间插值', '光谱线性回归', 'Transformer'], physical.values()))
            for col, (label, image) in enumerate(images, 1):
                im = axes[row, col].imshow(np.where(q, image, np.nan), vmin=lo, vmax=hi, cmap='viridis', interpolation='nearest')
                subtitle = '' if col == 1 else f"\nRMSE={np.sqrt(mse[list(normalized)[col-2]])*s['std']:.4f}"
                axes[row, col].set_title(label+subtitle)
                if p <= 3:
                    for iy, ix in np.argwhere(q):
                        axes[row, col].text(ix, iy, f'{image[iy,ix]:.3f}', ha='center', va='center', fontsize=8,
                                            color='white' if image[iy,ix] < (hi+lo)/2 else 'black')
            fig.colorbar(im, ax=axes[row, 1:5].tolist(), shrink=.65, label='地表反射率', pad=.01)
            limit = max(float(np.abs(v-target)[q].max()) for v in physical.values()) or 1e-5
            for col, (label, image) in enumerate(zip(['插值误差', '回归误差', '模型误差'], physical.values()), 5):
                err = axes[row, col].imshow(np.where(q, image-target, np.nan), vmin=-limit, vmax=limit, cmap='RdBu_r', interpolation='nearest')
                axes[row, col].set_title(label)
            fig.colorbar(err, ax=axes[row, 5:].tolist(), shrink=.65, label='预测−真实', pad=.01)
            for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
            name = f'{sensor}_{row}'
            np.savez_compressed(destination/f'{name}.npz', truth=target, valid=q, **physical)
            evidence.append(dict(sensor=sensor, band=b.name, observation_id=r['observation_id'], date=r['date'],
                                 mask_patch_yx=[y,x], native_gsd_m=b.native_gsd_m, mse=mse,
                                 arrays=f'{name}.npz', checkpoint_step=state['step']))
        fig.suptitle(f'{sensor} 地域留出：固定60m缺口；每波段按冻结清单取首例，未按效果挑选\n左为完整原图定位；右侧仅放大隐藏区，不将已观测像元充作预测。灰色/空白为无效区。', fontsize=13)
        fig.savefig(destination/f'{sensor}_comparison.png', dpi=150)
        plt.close(fig)
    metadata = dict(selection='first frozen VAL member for each predeclared band, no performance selection',
                    checkpoint_sha256=hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest(),
                    checkpoint_step=state['step'], cases=evidence, training_updates=0)
    (destination/'cases.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='complete', cases=len(evidence), output=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
