"""Full-coverage shuffled epochs; native observations need not share tensor shapes."""
import math
import torch
from torch.utils.data import BatchSampler, SubsetRandomSampler


def family(row):
    return 'Landsat' if row['sensor'] in ('L8','L9') else row['sensor']


class EpochBatches:
    def __init__(self, rows, seed, batch_size):
        if type(batch_size) is not int or batch_size<1:
            raise ValueError('Positive integer batch_size required')
        self.rows=sorted((r for r in rows if r['role']=='TRAIN' and len(r['bands'])>=2),key=lambda r:r['id'])
        if not self.rows or len({r['id'] for r in self.rows})!=len(self.rows):
            raise ValueError('Nonempty, unique TRAIN observations required')
        self.seed=seed;self.batch_size=batch_size;self.epoch=None;self.batches=None

    def __len__(self):return math.ceil(len(self.rows)/self.batch_size)

    def batch(self, completed_batches):
        if type(completed_batches) is not int or completed_batches<0:
            raise ValueError('Nonnegative integer batch cursor required')
        epoch,offset=divmod(completed_batches,len(self))
        if self.epoch!=epoch:
            generator=torch.Generator().manual_seed(self.seed+epoch)
            sampler=SubsetRandomSampler(range(len(self.rows)),generator=generator)
            self.batches=list(BatchSampler(sampler,self.batch_size,drop_last=False));self.epoch=epoch
        return [(family(self.rows[i]),self.rows[i]) for i in self.batches[offset]]

    def observations_seen(self, completed_batches):
        epoch,offset=divmod(completed_batches,len(self))
        return epoch*len(self.rows)+min(offset*self.batch_size,len(self.rows))

    def progress(self, completed_batches):
        return self.observations_seen(completed_batches)/len(self.rows)

    def validation_due(self, completed_batches, checks_per_epoch):
        if type(checks_per_epoch) is not int or checks_per_epoch<1:
            raise ValueError('Positive validations_per_epoch required')
        count=min(checks_per_epoch,len(self));position=(completed_batches-1)%len(self)+1
        return completed_batches>0 and position in {math.ceil(i*len(self)/count) for i in range(1,count+1)}
