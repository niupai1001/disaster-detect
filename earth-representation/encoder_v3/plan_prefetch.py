"""One bounded CPU geometry lookahead; only committed updates advance replay."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from .data import Observation,to_device
from .tasks import Target


def move_views(views,device):
    result=[]
    for view in views:
        task=view.task
        visible=to_device(Observation(task.visible,task.metadata),device).bands
        targets=to_device(Observation([t.band for t in task.targets],task.metadata),device).bands
        moved=replace(task,visible=visible,targets=[Target(b,t.mask.to(device)) for b,t in zip(targets,task.targets,strict=True)])
        result.append(replace(view,task=moved))
    return result


class PlanPrefetch:
    def __init__(self,prepare,*,start,stop,identity,prefetch):
        if type(start) is not int or type(stop) is not int or not 0<=start<=stop:raise ValueError('Invalid committed interval')
        self.prepare=prepare;self.next_update=start;self.stop=stop;self.identity=identity;self.prefetch=prefetch;self.delivered=None;self.running=False
    def state_dict(self):return dict(identity=self.identity,next_update=self.next_update)
    def commit(self,update):
        if not self.running or self.delivered!=update or update!=self.next_update:raise ValueError('Only delivered update can commit')
        self.next_update+=1;self.delivered=None
    def __iter__(self):
        if self.running:raise RuntimeError('Only one consuming iterator allowed')
        self.running=True;pool=ThreadPoolExecutor(max_workers=1) if self.prefetch else None;future=None
        try:
            for index in range(self.next_update,self.stop):
                if index!=self.next_update:raise RuntimeError('Commit before requesting another update')
                value=future.result() if future is not None else self.prepare(index)
                future=pool.submit(self.prepare,index+1) if pool is not None and index+1<self.stop else None
                self.delivered=index;yield index,value
                if self.delivered is not None:raise RuntimeError('Consumed update must commit')
        finally:
            if future is not None:future.cancel()
            if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
            self.running=False;self.delivered=None
