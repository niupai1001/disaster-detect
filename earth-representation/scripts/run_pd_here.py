"""Prepare and verify the independent PD-HERE V1 experiment."""
import argparse
import json
from pathlib import Path
import ssl  # Load stdlib ssl before adding the project's src/ssl to the import path.
import sys


def baseline(root, out, device):
    import math
    import torch

    release = root / 'outputs/encoder_v3/patch_latent/mit_global_ab30_v1/release'
    sys.path[:0] = [str(release / 'runtime/scripts'), str(release / 'runtime/src')]
    import train_patch_latent as runner

    manifest = runner.read(release / 'source_manifest.json')
    if any(runner.sha256(release / 'runtime' / name) != value for name, value in manifest.items()):
        raise ValueError('Historical frozen runtime does not match its manifest')
    binding = runner.read(release / 'identity.json')
    cfg = runner.read(release / 'config.json')
    settings = dict(cfg['training'], device=device, offload_native_values=float('inf'))
    torch.set_num_threads(settings['threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.device(device).type == 'cuda':
        torch.cuda.set_device(device)
        torch.zeros(1, device=device)
    data, resolver = runner.load_data()
    payload = torch.load(release / 'best.pt', map_location='cpu', weights_only=False)
    if payload['identity'] != binding:
        raise ValueError('Historical checkpoint binding mismatch')
    actual = runner.identity(cfg, data, resolver)
    if any(actual[k] != binding[k] for k in ('data', 'members', 'config', 'physical_catalog')):
        raise ValueError('Historical data and configuration do not match the frozen run')
    model = runner.Pretraining(runner.Config(**cfg['model']))
    model.load_state_dict(payload['model'])
    model.to(device)
    print(f'Replaying historical checkpoint on {len(data["VAL"].rows)} VAL observations', flush=True)
    result = runner.validate(model, data['VAL'], resolver, settings, lambda: False)
    stored = runner.read(release / f'validation_epoch_{payload["runner"]["best_epoch"]:02}.json')
    match = result['observations'] == stored['observations']
    for kind in ('A', 'B1'):
        for metric in ('mse', 'texture_gradient_mse'):
            match &= math.isclose(result['tasks'][kind][metric], stored['tasks'][kind][metric],
                                  abs_tol=5e-6, rel_tol=1e-5)
    from encoder_v3.shared_mae.train import write
    out.mkdir(parents=True, exist_ok=True)
    write(out / 'baseline.json', dict(status='PASS' if match else 'MISMATCH',
          run=runner.read(release / 'status.json')['run_id'], expected=stored['tasks'], **result))
    print(json.dumps(dict(match=match, tasks=result['tasks'], seconds=result['seconds'])), flush=True)
    if not match:
        raise SystemExit(1)


def main():
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=default_root)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--config', type=Path, default=default_root / 'configs/config_pd_here_v1.json')
    parser.add_argument('--device', default='cuda:1')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'baseline', 'capacity', 'recovery'):
        commands.add_parser(name)
    fit = commands.add_parser('fit')
    fit.add_argument('task', choices=('A', 'B1', 'joint'))
    fit.add_argument('--steps', type=int, default=200)
    fit.add_argument('--resume', action='store_true')
    fit.add_argument('--stop-after', type=int)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('checkpoint', type=Path)
    evaluate.add_argument('--split', choices=('VAL', 'TEST'), default='VAL')
    evaluate.add_argument('--test-selected-on-val', action='store_true')
    commands.add_parser('reference', help='Evaluate IDW, nearest and TRAIN-only spectral regression on VAL')
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.out = (args.out or args.root / 'outputs/encoder_v4/pd_here_v1').resolve()
    if args.command == 'baseline':
        baseline(args.root, args.out, args.device)
    else:
        sys.path[:0] = [str(default_root / 'src'), str(args.root / 'src')]
        from encoder_v4 import experiment
        experiment.run(args)


if __name__ == '__main__':
    main()
