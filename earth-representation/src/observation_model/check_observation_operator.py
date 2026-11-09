"""A fixed-observation counterexample: 30m data cannot uniquely determine 10m detail."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import ROOT
from .operators import observable_10_to_30


def main():
    out = ROOT/"observation_model/outputs/phase1/observation_operator"
    out.mkdir(parents=True, exist_ok=True)
    yy, xx = torch.meshgrid(torch.arange(120, device="cuda"), torch.arange(120, device="cuda"), indexing="ij")
    pattern = ((xx+yy) % 2).float()*2-1
    first = (.2+.04*pattern)[None].requires_grad_()
    second = (.2-.04*pattern)[None]
    valid = torch.ones_like(first, dtype=torch.bool)
    p10 = torch.tensor([[5., 5.]], device="cuda")
    p30 = torch.tensor([[15., 15.]], device="cuda")
    a, coverage = observable_10_to_30(first, valid, p10, p30)
    b, _ = observable_10_to_30(second, valid, p10, p30)
    # The interior excludes finite-window convolution boundary support.
    interior = (a-b)[0, 2:-2, 2:-2]
    rmse10 = float(((first-second)**2).mean().sqrt().detach())
    rmse30 = float(interior.square().mean().sqrt().detach())
    a.mean().backward()
    assert torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0
    assert rmse10 > .07 and rmse30 < .0002
    constant, _ = observable_10_to_30(torch.full_like(first, .2), valid, p10, p30)
    assert torch.allclose(constant, torch.full_like(constant, .2), atol=1e-6)
    arrays = [first[0].detach().cpu().numpy(), second[0].cpu().numpy(),
              a[0].detach().cpu().numpy(), b[0].cpu().numpy()]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    titles = ["10 m estimate A", "10 m estimate B", "A after Gaussian + area to 30 m", "B after Gaussian + area to 30 m"]
    for ax, array, title in zip(axes.flat, arrays, titles):
        picture = ax.imshow(array, cmap="viridis", vmin=.15, vmax=.25, interpolation="nearest")
        ax.set_title(title); ax.set_xlabel("Pixel column"); ax.set_ylabel("Pixel row")
    fig.colorbar(picture, ax=axes.ravel().tolist(), label="Surface reflectance", shrink=.7)
    fig.suptitle(f"10 m RMSE = {rmse10:.3f}; 30 m interior RMSE = {rmse30:.2g}")
    fig.savefig(out/"nonuniqueness.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(out/"counterexample.npz", first10=arrays[0], second10=arrays[1], first30=arrays[2], second30=arrays[3])
    result = dict(status="passed", operator="fixed7x7 Gaussian sigma10m; physical area sampling to30m",
                  rmse10=rmse10, rmse30_interior=rmse30, rmse30_full=float((a-b).square().mean().sqrt().detach()),
                  valid_gradient=True, constant_preserved=True, coverage_min=float(coverage.min()),
                  conclusion="different 10m patterns can agree with approximately the same30m observation",
                  boundary="synthetic identifiability demonstration; not evidence of training benefit or a calibrated OLI PSF")
    (out/"acceptance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
