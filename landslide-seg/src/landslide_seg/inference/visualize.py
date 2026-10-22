"""可视化：RGB、云/阴影、GT、粗概率、细概率、二值结果。

设计原则（避免"图骗人"）：
- 概率图**固定 vmin=0, vmax=1**，不做自动拉伸。否则当一个几乎恒定的概率场
  （例如 0.476–0.520）被 imshow 自动归一化时，微弱的周期纹理会被放大成明显网格，
  看起来像模型输出了结构，其实只是显示造成的假象。
- 每张子图标题都写出该图的真实数值范围，便于判断对比度是否被拉伸。
- 有 SCL 时额外画一张云/阴影图，直接回答"这些云是数据本身还是后加的"。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: 12 波段顺序下 RGB 合成用的通道（B04, B03, B02）
RGB_IDX = (3, 2, 1)


def _stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    a, b = np.percentile(band, lo), np.percentile(band, hi)
    if b - a < 1e-6:
        b = a + 1e-6
    return np.clip((band - a) / (b - a), 0, 1)


def rgb_from_bands(x: np.ndarray, idx=RGB_IDX) -> np.ndarray:
    """``[C,H,W]`` -> ``[H,W,3]`` float [0,1]（逐通道 2–98 分位拉伸）。"""
    return np.stack([_stretch(x[i]) for i in idx], axis=-1)


def save_panel(
    path: str | Path,
    rgb: np.ndarray,
    gt: np.ndarray,
    coarse_prob: np.ndarray,
    fine_prob: np.ndarray,
    fine_mask: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    scl: np.ndarray | None = None,
    title: str = "",
    presence: float | None = None,
) -> Path:
    """保存多联图。

    Args:
        rgb: ``[H,W,3]`` 0–1
        gt: ``[H,W]`` {0,1}
        coarse_prob: ``[H,W]`` **粗阶段** sigmoid 概率
        fine_prob: ``[H,W]`` **细阶段** sigmoid 概率
        fine_mask: ``[H,W]`` 阈值化后的二值结果
        valid: 可选，``[H,W]`` bool
        scl: 可选，``[H,W]`` Sen12 的 SCL 类别（用于画出云/阴影）
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    panels: list[tuple] = [
        (rgb, "RGB (B04/B03/B02, 2-98% stretch)", None, None, None),
        (gt, "GT mask", "gray", 0, 1),
        (coarse_prob, f"coarse prob\n[{coarse_prob.min():.3f}, {coarse_prob.max():.3f}]",
         "viridis", 0, 1),
        (fine_prob, f"fine prob\n[{fine_prob.min():.3f}, {fine_prob.max():.3f}]",
         "viridis", 0, 1),
        (fine_mask.astype(float), "fine mask @0.5", "gray", 0, 1),
    ]
    if scl is not None:
        # 2=地形阴影 3=云影 8/9=云 10=薄云；仅用于展示，不参与训练
        cloud = np.isin(scl, (2, 3, 8, 9, 10))
        panels.insert(1, (cloud.astype(float),
                          f"SCL cloud/shadow {cloud.mean()*100:.1f}%", "Reds", 0, 1))
    if valid is not None:
        panels.append(((~valid).astype(float),
                       f"invalid px {(~valid).mean()*100:.1f}%", "Reds", 0, 1))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.7 * n, 4.3))
    if n == 1:
        axes = [axes]
    for ax, (img, name, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    sup = title
    if presence is not None:
        sup += f"    presence={presence:.3f}"
    fig.suptitle(sup, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
