"""Second-stage USE v2 continuation: a 10k cosine tail from the completed 20k state.

Additive entry point. `scripts/run_use_v2.py` and `configs/use_v2/execution.yaml`
stay byte-identical, so the running engineering_v2 jobs keep verifying; this
script points the shared loader at `configs/use_v2_tail` instead. The tail is a
two-stage schedule, not a single 30k budget: the encoder and reconstruction head
start from `u20000.pt`, a fresh AdamW and a new 10,000-step cosine are used, and
the global step counter continues at 20,001 so the sample/target streams never
replay the first 20k updates.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_use_v2 as runtime
from pretrain_use_v2 import initialize, deterministic, ssl_update, lr, atomic_save, checkpoint, state_sha256
from evaluate_use_v2 import evaluate, selection

CONTRACT = ROOT / 'configs/use_v2_tail'
TOTAL = 30000


def active_jobs_cuda_only():
    """A CPU engineering run must never look like a job occupying card 0."""
    result = []
    for path in runtime.BASE.rglob('process.json'):
        try:
            data = runtime.read(path)
        except (OSError, ValueError):
            continue
        device = str(data.get('device', 'cpu'))
        if not device.startswith('cuda'):
            continue
        if data.get('status') == 'RUNNING' and data.get('pid') and runtime.card_process_alive(int(data['pid'])):
            result.append((path.parent.relative_to(runtime.ROOT).as_posix(), int(torch.device(device).index or 0)))
    return result


@contextmanager
def tail_contract():
    """Point the shared CLI at the tail contract only while a tail call runs, then restore it."""
    saved = (runtime.CONTRACT, runtime.SCRIPTS, runtime.matrix, runtime.allocation, runtime.waves,
             runtime.source_checkpoint, runtime.active_jobs)
    runtime.CONTRACT = CONTRACT
    if 'run_use_v2_tail.py' not in runtime.SCRIPTS:
        runtime.SCRIPTS = tuple(runtime.SCRIPTS) + ('run_use_v2_tail.py',)
    runtime.matrix, runtime.allocation, runtime.waves = tail_jobs, tail_allocation, tail_waves
    runtime.source_checkpoint = source_checkpoint
    runtime.active_jobs = active_jobs_cuda_only
    try:
        yield
    finally:
        (runtime.CONTRACT, runtime.SCRIPTS, runtime.matrix, runtime.allocation, runtime.waves,
         runtime.source_checkpoint, runtime.active_jobs) = saved


def source_checkpoint(job, engineering=False):
    """Same strict checks as the 20k path, but the tail source must report 30,000 updates."""
    if engineering or not job['source']:
        return None
    path = runtime.BASE / 'runs' / job['source']
    try:
        summary = runtime.read(path.parent / 'summary.json')
    except (OSError, ValueError):
        raise ValueError('Tail source pretraining has not produced a summary yet: ' + str(job['source']))
    if not summary['complete'] or summary['updates'] != TOTAL:
        raise ValueError('Tail source pretraining incomplete')
    if runtime.sha(path) != summary['checkpoints'][path.name]:
        raise ValueError('Tail source checkpoint changed')
    return path



def tail(candidate, out, device, steps=10000, gate=None, release_hash=None):
    with tail_contract():
        return _tail(candidate, out, device, steps, gate, release_hash)


def _tail(candidate, out, device, steps=10000, gate=None, release_hash=None):
    config, cal, splits, stats = runtime.load_configuration()
    spec = config['use_v2']['tail']
    if steps != spec['steps'] and steps > 100:
        raise ValueError('Only the full tail or an engineering smoke (<=100 steps) may run')
    deterministic()
    model = initialize(candidate, device)
    source = runtime.BASE / 'runs' / spec['source'][candidate]
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model.use.load_state_dict(saved['use'], strict=True)
    if state_sha256(model.use) != saved['use_sha256']:
        raise ValueError('Tail did not start exactly from the 20k encoder state')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=spec['lr'], weight_decay=.01)
    identity = dict(job_id=f'pretrain_{candidate}_tail', job=spec, release=release_hash,
                    engineering=steps <= 100,
                    config_sha256=runtime.sha(CONTRACT / 'execution.yaml'),
                    calibration_sha256=runtime.sha(CONTRACT / 'calibration.json'),
                    initialization_sha256=state_sha256(model),
                    source_sha256={**{name: runtime.sha(ROOT / 'scripts' / name) for name in runtime.SCRIPTS},
                                   'models/use_v2.py': runtime.sha(ROOT / 'src/landslide_seg/models/use_v2.py')},
                    source_checkpoint_sha256=runtime.sha(source),
                    tail_source=str(source.relative_to(ROOT)))
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    runtime.write_json(out / 'resolved_config.json', dict(configuration=config, job=spec, identity=identity,
                                                          device_gate=gate, command=sys.argv, cwd=str(Path.cwd())), True)
    offset = spec['global_step_offset']
    atomic_save(checkpoint(model, optimizer, offset, float('inf'), identity, []), out / 'initialization.pt')
    runtime.write_json(out / 'process.json', dict(pid=__import__('os').getpid(), command=sys.argv, device=device,
                                                  device_gate=gate, started=time.time(), status='RUNNING'))
    began = time.perf_counter(); history = []; best = float('inf'); last_saved = offset; local = 0
    try:
        for local in range(1, steps + 1):
            step = offset + local
            for group in optimizer.param_groups:
                group['lr'] = lr(local, steps, spec['warmup'])
            model.train()
            record = ssl_update(model, optimizer, splits['TRAIN'], stats, step, cal, None, device)
            record.update(step=step, lr=optimizer.param_groups[0]['lr'])
            history.append(record)
            with (out / 'curve.jsonl').open('a', encoding='utf8') as handle:
                handle.write(json.dumps(record) + '\n')
            validate = local % 1000 == 0 or local == steps
            if validate:
                evaluation = evaluate(model, splits['PRETRAIN_VAL'], stats, device, limit=None, calibration=cal)
                score = selection(evaluation, cal)
                runtime.write_json(out / f'validation_{step}.json', evaluation)
                record.update(selection_score=score, selection_improved=bool(score < best))
                if score < best:
                    best = score
                    atomic_save(checkpoint(model, optimizer, step, best, identity, history), out / f'selected_{step}.pt')
                    runtime.atomic_copy(out / f'selected_{step}.pt', out / 'best_unlabeled.pt')
            if local % 1000 == 0 or validate:
                atomic_save(checkpoint(model, optimizer, step, best, identity, history), out / 'last.pt')
                last_saved = step
            if local in (steps // 2, steps):
                atomic_save(checkpoint(model, optimizer, step, best, identity, history), out / f'u{step}.pt')
        if last_saved < offset + local:
            atomic_save(checkpoint(model, optimizer, offset + local, best, identity, history), out / 'last.pt')
        (out / 'curve.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in history), encoding='utf8')
        elapsed = time.perf_counter() - began
        check = initialize(candidate, device)
        check.use.load_state_dict(saved['use'], strict=True)
        summary = dict(complete=local == steps, updates=offset + local, formal_updates=0 if steps <= 100 else local,
                       engineering_updates=local if steps <= 100 else 0, best_score=best, identity=identity,
                       seconds_this_session=elapsed, seconds_per_update=elapsed / max(1, local),
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if 'cuda' in str(device) else None,
                       parameters=sum(p.numel() for p in model.parameters()),
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       checkpoints={p.name: runtime.sha(p) for p in out.glob('*.pt')},
                       initialization_reset_sha256=state_sha256(check),
                       tail=dict(source=identity['tail_source'], steps=steps, warmup=spec['warmup'], lr=spec['lr']))
        if summary['initialization_reset_sha256'] != identity['initialization_sha256']:
            raise ValueError('Tail initialization rebuild mismatch')
        runtime.write_json(out / 'summary.json', summary)
        runtime.write_json(out / 'process.json', dict(pid=__import__('os').getpid(), status='COMPLETED' if summary['complete'] else 'STOPPED'))
        return summary
    except BaseException as error:
        runtime.write_json(out / 'failure.json', dict(error=repr(error), successful_updates=len(history),
                                                      saved_step=last_saved, command=sys.argv))
        raise


def smoke(out, device='cpu'):
    with tail_contract():
        return _smoke(out, device)


def _smoke(out, device='cpu'):
    """Engineering smoke: exact start state, stream offset, schedule endpoints, one probe and one downstream step."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    config, cal, splits, stats = runtime.load_configuration()
    spec = config['use_v2']['tail']
    assert lr(spec['warmup'], spec['steps'], spec['warmup']) == spec['lr']
    assert lr(spec['steps'], spec['steps'], spec['warmup']) < 1e-9
    source = runtime.BASE / 'runs' / spec['source']['mean']
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model = initialize('mean', 'cpu'); model.use.load_state_dict(saved['use'], strict=True)
    assert state_sha256(model.use) == saved['use_sha256']
    from pretrain_use_v2 import sample_rows
    first = sample_rows(splits['TRAIN'], spec['global_step_offset'] + 1)
    assert [r['file'] for r in first] == [r['file'] for r in sample_rows(splits['TRAIN'], spec['global_step_offset'] + 1)]
    gone = sample_rows(splits['TRAIN'], spec['global_step_offset'] + 1)
    assert [r['file'] for r in gone] != [r['file'] for r in sample_rows(splits['TRAIN'], 1)]
    receipt = dict(passed=True, tail_steps=20, device=device,
                   start_state_equals_20k=saved['use_sha256'],
                   schedule=dict(peak=spec['lr'], end=float(lr(spec['steps'], spec['steps'], spec['warmup'])),
                                 warmup=spec['warmup'], steps=spec['steps']),
                   stream_offset=spec['global_step_offset'])
    summary = tail('mean', out / 'tail_mean_smoke', 'cpu', steps=20)
    assert summary['updates'] == spec['global_step_offset'] + 20
    receipt['tail_smoke'] = dict(updates=summary['updates'], best=summary['best_score'],
                                 checkpoints=summary['checkpoints'])
    for job in ('probe_tail_mean_fine', 'downstream_tail_mean_context'):
        result = runtime.train(job, out / job, 'cpu', engineering=True, engineering_steps=2, gate=dict(device='cpu'))
        receipt[job] = dict(updates=result['updates'], complete=result['complete'])
    runtime.write_json(out / 'acceptance.json', receipt, True)
    return receipt



def tail_jobs():
    return yaml.safe_load((CONTRACT / "execution.yaml").read_text(encoding="utf8"))["use_v2"]["jobs"]


def tail_allocation():
    """Attention family on cuda:0, mean family on cuda:1; old-USE jobs split one each."""
    plan = {}
    for job in tail_jobs():
        if "attention" in job:
            plan[job] = "cuda:0"
        elif "mean" in job:
            plan[job] = "cuda:1"
        else:
            plan[job] = "cuda:0" if job.endswith("old_context") else "cuda:1"
    steps = {card: sum(tail_jobs()[j]["steps"] for j in plan if plan[j] == card) for card in ("cuda:0", "cuda:1")}
    assert steps == {"cuda:0": 17000, "cuda:1": 17000}, steps
    return plan


def tail_waves():
    plan = tail_allocation()
    return [dict(wave="evaluation", jobs={card: [j for j in plan if plan[j] == card] for card in ("cuda:0", "cuda:1")})]



def snapshot_tail(out, acceptance_path, marker):
    with tail_contract():
        return _snapshot_tail(out, acceptance_path, marker)


def _snapshot_tail(out, acceptance_path, marker):
    acceptance = json.loads((ROOT / acceptance_path).read_text(encoding="utf8"))
    assert acceptance["passed"] is True
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    suite = ROOT / acceptance["suite_directory"]
    files = [*sorted((ROOT / "src/landslide_seg").rglob("*.py")),
             *[ROOT / "scripts" / name for name in runtime.SCRIPTS],
             ROOT / "tests/test_use_v2.py", ROOT / "tests/test_use_device_gate.py",
             *sorted(CONTRACT.glob("*")), ROOT / acceptance_path, *sorted(suite.glob("*")),
             *[ROOT / "manifests/v4r1" / name for name in ("pool_100.jsonl", "stats_core25.json", "protocol.json")],
             ROOT / "outputs/use_independent_frozen/cache_audit_v1/acceptance.json",
             ROOT / "outputs/use_independent_frozen/cache_audit_v1/members.jsonl",
             ROOT / "outputs/use/pretraining/resolved_config.json", ROOT / "outputs/use/quality/index.json",
             ROOT / "outputs/use_independent_frozen/pretraining/multisensor/best.pt",
             ROOT / "outputs/use_independent_frozen/pretraining/multisensor/summary.json",
             ROOT / "docs/assets/research/research_policy.json",
             runtime.BASE / "runs/pretrain_mean/u20000.pt", runtime.BASE / "runs/pretrain_attention/u20000.pt"]
    digests = {}
    for source in files:
        if source.is_dir():
            continue
        name = source.relative_to(ROOT).as_posix(); expected = runtime.sha(source)
        target = out / "snapshot" / name; target.parent.mkdir(parents=True, exist_ok=True)
        import shutil; shutil.copy2(source, target)
        if runtime.sha(target) != expected:
            raise ValueError("Changed during snapshot: " + name)
        digests[name] = expected
    device = {}
    for candidate in ("mean", "attention"):
        device[candidate] = dict(cuda="cuda:1" if candidate == "mean" else "cuda:0",
                                 out=f"outputs/use_v2/runs/pretrain_{candidate}_tail",
                                 steps=10000)
    commands = {job: [sys.executable, "-X", "utf8", "scripts/run_use_v2_tail.py", "train", "--job", job,
                      "--release", str(out), "--audit-marker", str(ROOT / marker), "--device", tail_allocation()[job]]
                for job in tail_jobs()}
    tail_commands = {candidate: [sys.executable, "-X", "utf8", "scripts/run_use_v2_tail.py", "tail",
                                 "--candidate", candidate, "--out", f"outputs/use_v2/runs/pretrain_{candidate}_tail",
                                 "--steps", "10000", "--release", str(out),
                                 "--audit-marker", str(ROOT / marker), "--device",
                                 "cuda:1" if candidate == "mean" else "cuda:0"] for candidate in ("mean", "attention")}
    import platform, importlib.metadata
    manifest = dict(status="READY_FOR_AUDIT", files=digests, created=time.time(),
                    engineering_acceptance=acceptance,
                    environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                     dependencies={n: importlib.metadata.version(n) for n in runtime.DEPENDENCIES}),
                    cwd=str(ROOT), formal_updates=0, commands=commands, tail_commands=tail_commands,
                    allocation=tail_allocation(), waves=tail_waves(), tail_budget=20000, eval_budget=34000,
                    stage="second-stage 10k cosine tail from the completed 20k state; two-stage schedule, not a single 30k budget",
                    resume="A tail run is restarted from u20000.pt (deterministic) rather than resumed; evaluation jobs use the same command with --resume",
                    limits=["Independent C audit required", "Two-stage schedule, not a 30,000-step single budget",
                            "Single seed; no significance claim"])
    runtime.write_json(out / "manifest.json", manifest, True)
    (out / "manifest.sha256").write_text(runtime.sha(out / "manifest.json") + "\n", encoding="utf8")
    runtime.verify_release(out)
    return dict(release=str(out), sha256=runtime.sha(out / "manifest.json"), status=manifest["status"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('tail', 'train', 'smoke', 'snapshot', 'verify'))
    parser.add_argument('--candidate', choices=('mean', 'attention'))
    parser.add_argument('--out'); parser.add_argument('--device', default='cpu'); parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--release'); parser.add_argument('--audit-marker'); parser.add_argument('--acceptance')
    parser.add_argument('--job', choices=tuple(yaml.safe_load((CONTRACT / 'execution.yaml').read_text(encoding='utf8'))['use_v2']['jobs']))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'tail':
        digest = runtime.verify_release(args.release, args.audit_marker) if args.steps > 100 else None
        with runtime.gpu_guard(args.device) as gate:
            result = tail(args.candidate, args.out, args.device, args.steps, gate, digest)
    elif args.action == 'train':
        digest = runtime.verify_release(args.release, args.audit_marker)
        with tail_contract(), runtime.gpu_guard(args.device) as gate:
            result = runtime.train(args.job, runtime.BASE / 'runs' / args.job, args.device,
                                   release_hash=digest, gate=gate)
    elif args.action == 'smoke':
        result = smoke(args.out, args.device)
    elif args.action == 'snapshot':
        result = snapshot_tail(args.out, args.acceptance, args.audit_marker)
    else:
        result = runtime.verify_release(args.release, args.audit_marker)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()





