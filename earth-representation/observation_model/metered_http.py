"""Seekable HTTP range access with an aggregate response-body download cap."""
import io
import json
import threading
from pathlib import Path

import requests


class DownloadBudget:
    def __init__(self,path,cap):
        self.path=Path(path);self.cap=cap;self.lock=threading.Lock()
        self.received=json.loads(self.path.read_text())["received_bytes"] if self.path.exists() else 0
        self.session=requests.Session()
        self.session.headers["Accept-Encoding"]="identity"

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps(dict(received_bytes=self.received,cap_bytes=self.cap,unit="HTTP response body bytes read, including metadata queries"),indent=2))

    def json_request(self,method,url,**kwargs):
        return json.loads(self.bytes_request(method,url,**kwargs))

    def bytes_request(self,method,url,**kwargs):
        with self.lock:
            remaining=self.cap-self.received
            if remaining<=0:raise RuntimeError("DOWNLOAD_BUDGET_EXHAUSTED")
            with self.session.request(method,url,stream=True,timeout=(20,60),**kwargs) as response:
                response.raise_for_status();data=response.raw.read(remaining)
                self.received+=len(data);self.save()
                return data

    def get(self,url,start,end):
        with self.lock:
            remaining=self.cap-self.received
            if end-start+1>remaining:raise RuntimeError("DOWNLOAD_BUDGET_EXHAUSTED")
            with self.session.get(url,headers={"Range":f"bytes={start}-{end}"},stream=True,timeout=(20,60)) as response:
                response.raise_for_status()
                if response.status_code!=206:raise ValueError("server did not honor bounded range request")
                data=response.raw.read(end-start+1)
                self.received+=len(data)
                self.save()
                total=int(response.headers["Content-Range"].split("/")[-1])
                return data,total

    def open(self,url):return HTTPRangeFile(url,self)


class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,budget):
        super().__init__();self.url=url;self.budget=budget;self.pos=0
        self.first,self.size=budget.get(url,0,0);self.blocks={}
        self.block_size=262144

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError("negative seek")
        return self.pos
    def read(self,size=-1):
        if size<0:size=self.size-self.pos
        end=min(self.pos+size,self.size);parts=[]
        while self.pos<end:
            index=self.pos//self.block_size
            if index not in self.blocks:
                begin=index*self.block_size
                self.blocks[index]=self.budget.get(self.url,begin,min(begin+self.block_size,self.size)-1)[0]
            block=self.blocks[index];offset=self.pos%self.block_size
            take=min(len(block)-offset,end-self.pos)
            if take<=0:raise IOError("short HTTP range body")
            parts.append(block[offset:offset+take]);self.pos+=take
        return b"".join(parts)
    def readinto(self,buffer):
        value=self.read(len(buffer));buffer[:len(value)]=value;return len(value)
