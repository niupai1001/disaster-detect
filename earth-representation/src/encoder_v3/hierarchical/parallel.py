"""Two independent native-data replicas, one global mean and one optimizer step."""
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor,wait
import copy
import torch


def partition_rows(rows):
    shards=[[],[]];loads=[0,0]
    # Schedule expensive observations first, but restore the original journal order.
    for row in sorted(rows,key=lambda r:sum(b['shape'][0]*b['shape'][1] for b in r['bands']),reverse=True):
        slot=min(range(2),key=lambda i:loads[i])
        shards[slot].append(row)
        loads[slot]+=sum(b['shape'][0]*b['shape'][1] for b in row['bands'])
    return shards


def device_context(device):
    return torch.cuda.device(device) if str(device).startswith('cuda') else nullcontext()


def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


@torch.no_grad()
def add_replica_gradients(model,replica):
    primary=[p for p in model.parameters() if p.requires_grad]
    secondary=[p for p in replica.parameters() if p.requires_grad]
    if len(primary)!=len(secondary):raise ValueError('Replica parameter mismatch')
    # A single host-staged transfer also works without peer access or NCCL on Windows.
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in secondary]).cpu()
    flat=flat.to(primary[0].device)
    offset=0
    for target,source in zip(primary,secondary):
        gradient=flat[offset:offset+target.numel()].view_as(target);offset+=target.numel()
        if source.grad is not None:
            if target.grad is None:target.grad=gradient.clone()
            else:target.grad.add_(gradient)


class TwoDeviceBatch:
    def __init__(self,model,dataset,resolver,config,*,secondary_data=None,validation_dataset=None,secondary_validation_dataset=None):
        from .train import load_data
        self.model=model;self.config=config;self.devices=config['devices']
        if model._layouts:raise ValueError('Construct replicas before any geometry is cached')
        self.replica=copy.deepcopy(model).to(self.devices[1])
        # Each reader owns its adapter, including EnMAP's mutable input LRU cache.
        if secondary_data is None:
            datasets,second_resolver=load_data(config);second_dataset=datasets['TRAIN']
            secondary_validation_dataset=datasets['VAL']
        else:second_dataset,second_resolver=secondary_data
        if [r['id'] for r in dataset.rows]!=[r['id'] for r in second_dataset.rows]:raise ValueError('Replica data membership mismatch')
        if getattr(dataset,'identities',None)!=getattr(second_dataset,'identities',None):raise ValueError('Replica data contract mismatch')
        if getattr(resolver,'catalog_sha',None)!=getattr(second_resolver,'catalog_sha',None):raise ValueError('Replica physical catalog mismatch')
        self.datasets=(dataset,second_dataset);self.resolvers=(resolver,second_resolver)
        self.validation_datasets=None
        if validation_dataset is not None:
            if secondary_validation_dataset is None:raise ValueError('Independent secondary validation adapter required')
            if [r['id'] for r in validation_dataset.rows]!=[r['id'] for r in secondary_validation_dataset.rows] or getattr(validation_dataset,'identities',None)!=getattr(secondary_validation_dataset,'identities',None):raise ValueError('Replica validation membership/contract mismatch')
            self.validation_datasets=(validation_dataset,secondary_validation_dataset)
        self.pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='v3-h-gpu')

    @torch.no_grad()
    def broadcast(self):
        self.replica.load_state_dict(self.model.state_dict())
        self.replica.train(self.model.training)
        synchronize(self.devices[1])

    def backward(self,rows,epoch,stop_requested=None):
        return self.compute(rows,epoch,self.config,True,self.datasets,stop_requested)

    def evaluate(self,rows,config):
        if self.validation_datasets is None or any(r['role']!='VAL' for r in rows):raise ValueError('Bound VAL adapters required')
        return self.compute(rows,0,config,False,self.validation_datasets)

    def compute(self,rows,epoch,config,backward,datasets,stop_requested=None):
        from .train import batch_loss
        self.broadcast()
        if backward:self.model.zero_grad(set_to_none=True);self.replica.zero_grad(set_to_none=True)
        shards=partition_rows(rows)
        def work(slot):
            if not shards[slot]:return 0.,[]
            model=(self.model,self.replica)[slot];device=self.devices[slot]
            with device_context(device),torch.set_grad_enabled(backward):
                result=batch_loss(model,datasets[slot],shards[slot],self.resolvers[slot],
                    dict(config,device=device),epoch=epoch,backward=backward,
                    stop_requested=stop_requested,normalization_observations=len(rows))
                synchronize(device)
                return result
        futures=[self.pool.submit(work,i) for i in range(2)]
        wait(futures)  # Never mutate/save the master while another replica is still computing.
        results=[future.result() for future in futures]
        if backward:
            add_replica_gradients(self.model,self.replica)
            synchronize(self.devices[0])
        records={r['id']:r for _,part in results for r in part}
        if len(records)!=len(rows):raise ValueError('Global batch must contain each observation exactly once')
        ordered=[records[r['id']] for r in rows]
        return sum(r['loss'] for r in ordered)/len(rows),ordered

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
