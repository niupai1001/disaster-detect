"""Benchmark and use two model replicas for one logical update."""
from concurrent.futures import ThreadPoolExecutor, wait
import copy
import math
import random
import time

import torch

from encoder_v3.hierarchical.parallel import add_replica_gradients
from .whole_scene_batch import batched_backward


class PackedReplica:
    def __init__(self, model, cpu, gpu, signatures, masks, primary, secondary, second_cache):
        self.model = model
        self.replica = copy.deepcopy(model).to(secondary)
        self.devices = (primary, secondary)
        self.caches = (gpu, second_cache)
        self.cpu, self.signatures, self.masks = cpu, signatures, masks
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.cost = {key: scene.values.numel() + 32 * len(scene.values) for key, scene in cpu.items()}

    def backward(self, group, epoch):
        if len(group) < 2:
            return batched_backward(self.model, group, self.cpu, self.caches[0], self.signatures, self.masks, epoch)
        self.replica.load_state_dict(self.model.state_dict())
        self.replica.train(self.model.training)
        self.replica.zero_grad(set_to_none=True)
        torch.cuda.synchronize(self.devices[1])
        cap = math.ceil(len(group)/2)
        shards = [[], []]
        work = [0, 0]
        for row in sorted(group, key=lambda r: self.cost[r['id']], reverse=True):
            slot = min(range(2), key=lambda i: (len(shards[i])>=cap, work[i], i))
            shards[slot].append(row)
            work[slot] += self.cost[row['id']]

        def run(index, model):
            device = self.devices[index]
            with torch.cuda.device(device):
                losses = batched_backward(model, shards[index], self.cpu, self.caches[index],
                                          self.signatures, self.masks, epoch, parent_count=len(group))
                torch.cuda.synchronize(device)
                return losses
        jobs = [self.pool.submit(run, i, m) for i,m in enumerate((self.model,self.replica))]
        wait(jobs)
        a,b = [job.result() for job in jobs]
        add_replica_gradients(self.model, self.replica)
        return a + b.to(self.devices[0])

    def close(self):
        self.pool.shutdown(wait=True)


def tune_packed_parallel(model, rows, batch, cpu, gpu, signatures, masks, primary, secondary):
    """Require speed and equivalent gradients before using the second GPU."""
    if primary == secondary:
        raise ValueError('Parallel device must differ from primary device')
    if torch.device(secondary).index is None or torch.device(secondary).index >= torch.cuda.device_count():
        return None, dict(selected=[primary], reason='second device unavailable')
    ordered = rows.copy()
    random.Random(18).shuffle(ordered)
    batches = [ordered[i:i+batch] for i in range(0, min(7*batch,len(ordered)), batch)]
    twin = None

    def run(dual):
        times = []
        loss, grads = None, None
        for index, group in enumerate(batches):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize(primary)
            start = time.perf_counter()
            output = twin.backward(group,0) if dual else batched_backward(model,group,cpu,gpu,signatures,masks,0)
            torch.cuda.synchronize(primary)
            if index:
                times.append(time.perf_counter()-start)
            if index == 0:
                loss = output.cpu()
                grads = {k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
        return sum(times),loss,grads

    try:
        sample = {r['id']:cpu[r['id']].to(secondary) for group in batches for r in group}
        twin = PackedReplica(model,cpu,gpu,signatures,masks,primary,secondary,sample)
        serial,loss0,grad0 = run(False)
        parallel,loss1,grad1 = run(True)
        assert grad0.keys()==grad1.keys()
        error = sum(float((g-grad1[k]).square().sum()) for k,g in grad0.items())
        energy = sum(float(g.square().sum()) for g in grad0.values())
        relative_grad = (error/max(energy,1e-30))**.5
        relative_loss = float((loss0-loss1).norm()/loss0.norm().clamp_min(1e-12))
        if not (math.isfinite(relative_grad) and math.isfinite(relative_loss)):
            raise RuntimeError('Packed replica comparison was not finite')
        speed_ok = bool(parallel<.90*serial)
        equivalent = bool(relative_grad<.02 and relative_loss<.005)
        enabled = equivalent and speed_ok
        result = dict(selected=[primary,secondary] if enabled else [primary], compared_batches=len(batches)-1,
                      single_seconds=serial, dual_seconds=parallel, gradient_relative_l2=relative_grad,
                      loss_relative_l2=relative_loss)
        if enabled:
            copy_cache = {key:(sample[key] if key in sample else scene.to(secondary)) for key,scene in cpu.items()}
            twin.caches = (gpu,copy_cache)
            return twin,result
        twin.close()
        return None,result
    except (RuntimeError, AssertionError) as error:
        if twin is not None:
            twin.close()
        return None,dict(selected=[primary], reason=f'dual device preflight: {error}')
    finally:
        model.zero_grad(set_to_none=True)
