"""One bounded engineering update per view, never a formal training launcher."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from encoder_v3.latent_response.types import NativeGrid
from encoder_v3.shared_mae.data import load_data, sample_from_dataset
from encoder_v3.shared_mae.metadata import PhysicalChannel
from encoder_v3.shared_mae.model import NativeChannel, Sample

from .attention import Metrics
from .geometry import middle_map, overlap_map
from .model import Config, Decoder
from .coarse_shared import SharedCoarseDecoder, regional_supports
from .tasks import Pretraining, legacy_a_layout

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT/"configs/config_v3_h_revised_engineering_v1.json"


def synthetic(device, bands=4, shape=(16, 17)):
    grid = NativeGrid((10., 0., 500003., 0., -10., 5000101.), shape, "EPSG:32632")
    row, col = torch.meshgrid(torch.arange(shape[0], device=device),
                              torch.arange(shape[1], device=device), indexing="ij")
    base = (row + col) / 16
    return Sample("v3-h-r-synthetic", "TRAIN", [
        NativeChannel(base + i/10, torch.ones_like(base, dtype=torch.bool), grid,
                      PhysicalChannel("optical", "surface_reflectance", 10., 10.,
                                      center_um=.45+i*.004, source=f"synthetic:{i}"), str(i))
        for i in range(bands)])


def hashes(config):
    lr = json.loads((ROOT/"src/encoder_v3/config_optical_selfsup_v1.json").read_text(encoding="utf-8"))
    enmap = json.loads((ROOT/"configs/config_lr_e0_v1.json").read_text(encoding="utf-8"))
    optical = Path(lr["data"]["release"])
    hyperspectral = Path(enmap["enmap_release"])
    files = [config, optical/"members.jsonl", optical/"statistics.json",
             hyperspectral/"members.jsonl", hyperspectral/"statistics.json",
             ROOT/"src/encoder_v3/hierarchical/model.py", ROOT/"src/encoder_v3/hierarchical/geometry.py",
             ROOT/"src/encoder_v3/tasks.py", ROOT/"data/physics/sensor_physics.json"]
    files.extend(sorted(Path(__file__).parent.glob("*.py")))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def real_case(sensor, rank, device):
    datasets, resolver = load_data()
    train = datasets["TRAIN"]
    choices = [(sum(b["shape"][0]*b["shape"][1] for b in row["bands"]), row["id"], i)
               for i, row in enumerate(train.rows) if row["sensor"] == sensor]
    choices.sort()
    if not choices:
        raise ValueError("No matching sensor")
    _, _, i = choices[len(choices)//2 if rank == "median" else -1]
    return sample_from_dataset(train, i, resolver, device)


def profile(sample, cfg, settings, device, task, repeat, offload=False, variant="per_pixel"):
    torch.manual_seed(settings["seed"])
    decoder = {"per_pixel": Decoder, "shared_coarse": SharedCoarseDecoder}[variant]
    model = Pretraining(cfg, decoder_class=decoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"],
                                  weight_decay=settings["weight_decay"])
    phases = []
    for index in range(repeat):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        metrics = Metrics()
        phase = dict(cache_view="first_in_process" if index == 0 else "warm")
        def timed(label, call):
            cuda = device.type == "cuda"
            begin = torch.cuda.Event(enable_timing=True) if cuda else None
            end = torch.cuda.Event(enable_timing=True) if cuda else None
            start = time.perf_counter()
            if cuda:
                begin.record()
            value = call()
            if cuda:
                end.record()
                end.synchronize()
            phase[label] = dict(wall_seconds=time.perf_counter()-start,
                                cuda_elapsed_ms=begin.elapsed_time(end) if cuda else None)
            print(f"{sample.identifier} {task} #{index+1} {label} {phase[label]['wall_seconds']:.3f}s",
                  flush=True)
            return value
        context = torch.autograd.graph.save_on_cpu() if offload else torch.enable_grad()
        linear_flops = {}
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def count(layer, inputs, result, name=name):
                    linear_flops[name] = linear_flops.get(name, 0) + 2*result.numel()*layer.in_features
                hooks.append(module.register_forward_hook(count))
        try:
            with context:
                result = timed("forward", lambda: model(sample, kind=task, seed=settings["seed"], metrics=metrics))
        finally:
            for hook in hooks:
                hook.remove()
        native = result.encoded.native
        active = [i for i, ids in enumerate(native.ids) if len(ids)]
        convolution = 0
        for i in active:
            h, w = native.grids[i].shape
            halo_rows = 2*((h-1)//cfg.conv_tile_rows)
            convolution += 2*9*cfg.fine*(h+halo_rows)*w
        phase["flops_forward_multiply_add"] = dict(
            linear=sum(linear_flops.values()),
            descriptor_and_ffn={k: v for k, v in linear_flops.items() if "descriptor" in k or "ff" in k},
            attention_pairs=sum(row["pair_flops"] for row in metrics.attention.values()),
            dynamic_convolution=convolution,
            basis_mixing=2*len(active)*cfg.bases*cfg.fine*10,
            decoder_condition=metrics.attention['decoder_local'].get('condition_flops',0),
            dense_padding=sum(row.get('padding_flops',0) for row in metrics.attention.values()))
        if not torch.isfinite(result.loss):
            raise ValueError("Nonfinite loss")
        phase["loss"] = float(result.loss.detach())
        phase["channel_mse"] = {str(k): float(v.detach()) for k, v in result.channel_loss.items()}
        loss = result.loss
        del result
        timed("backward", loss.backward)
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Nonfinite gradient")
        timed("optimizer", optimizer.step)
        phase.update(metrics.summarize())
        phase["geometry"] = metrics.geometry
        phase["cache"] = {name: func.cache_info()._asdict()
                          for name, func in (("middle", middle_map), ("cross", overlap_map),
                                             ("legacy_mask", legacy_a_layout),
                                             ("decoder_regional", regional_supports))}
        if device.type == "cuda":
            phase["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            phase["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        phases.append(phase)
    return dict(task=task, phases=phases, parameters=sum(p.numel() for p in model.parameters()),
                optimizer_state_bytes=sum(t.numel()*t.element_size() for state in optimizer.state.values()
                                          for t in state.values() if isinstance(t, torch.Tensor)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("synthetic", "real"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", choices=("A", "B1"), required=True)
    parser.add_argument("--sensor", choices=("S2", "L8", "L9", "EnMAP"))
    parser.add_argument("--rank", choices=("median", "max"), default="median")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--geometry-cache", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or (args.mode == "real" and args.sensor is None):
        parser.error("Expected positive repeat and a sensor for real mode")
    if args.output.exists():
        parser.error("Choose a new output file")
    document = json.loads(args.config.read_text(encoding="utf-8"))
    if document["architecture"] != "hierarchical_v3_h_revised" or document["formal_training_authorized"] is not False:
        parser.error("Engineering-only V3-H-R configuration required")
    if args.geometry_cache is not None:
        os.environ["V3_H_R_GEOMETRY_CACHE"] = str(args.geometry_cache.resolve())
    if document["engineering"]["precision"] != "float32" or document["engineering"]["optimizer"] != "AdamW":
        parser.error("This engineering entry point measures FP32 AdamW")
    variant = document.get("decoder_variant", "per_pixel")
    if variant not in ("per_pixel", "shared_coarse"):
        parser.error("Unknown decoder variant")
    cfg = Config(**document["model"])
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    report = dict(mode=args.mode, sensor=args.sensor, rank=args.rank, device=str(device), decoder_variant=variant,
                  torch_version=torch.__version__, source_sha256=hashes(args.config),
                  cuda_device=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  activation_offload=args.offload, persistent_geometry_cache=args.geometry_cache is not None,
                  status="FAIL")
    try:
        start = time.perf_counter()
        sample = synthetic(device) if args.mode == "synthetic" else real_case(args.sensor, args.rank, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report.update(sample=sample.identifier, read_wall_seconds=time.perf_counter()-start,
                      native_values=sum(c.values.numel() for c in sample.channels),
                      channels=len(sample.channels))
        report.update(profile(sample, cfg, document["engineering"], device, args.task, args.repeat, args.offload, variant),
                      status="PASS")
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
        print(args.output, report["status"], flush=True)


if __name__ == "__main__":
    main()
