"""Local derived geometry cache, keyed by algorithm and exact spatial metadata."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch


def checksum(value):
    matrix=value[0]
    digest=hashlib.sha256(json.dumps(value[1:],sort_keys=True).encode())
    digest.update(matrix.indices().numpy().tobytes());digest.update(matrix.values().numpy().tobytes())
    return digest.hexdigest()


class DiskGeometry:
    def __init__(self,compute,root,algorithm_sha256):
        self.compute=compute;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.algorithm=algorithm_sha256;self.hits=0;self.misses=0;self.invalid=0
    def __call__(self,asset,grid=None):
        key=json.dumps([self.algorithm,asset.sample_grid['crs'],list(asset.sample_grid['transform']),list(asset.image.shape[-2:]),grid],sort_keys=True)
        path=self.root/(hashlib.sha256(key.encode()).hexdigest()+'.pt')
        if path.exists():
            try:
                saved=torch.load(path,map_location='cpu',weights_only=True);value=saved['value']
                if saved['key']!=key or checksum(value)!=saved['checksum']:raise ValueError('Invalid geometry cache')
                self.hits+=1;return value
            except Exception:
                self.invalid+=1
        self.misses+=1;value=self.compute(asset,grid)
        temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        torch.save(dict(key=key,value=value,checksum=checksum(value)),temporary)
        os.replace(temporary,path)
        return value
