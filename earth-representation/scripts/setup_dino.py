"""Fetch only the pinned official DINO sources required by this project."""
import hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    source=json.loads((ROOT/"configs/dino_source.json").read_text())
    destination=ROOT/"vendor/dino";destination.mkdir(parents=True,exist_ok=True)
    for name,expected in source["files"].items():
        path=destination/name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise RuntimeError(f"local {path} differs from pinned source; preserve and review it before replacement")
        else:
            url="https://raw.githubusercontent.com/facebookresearch/dino/"+source["commit"]+"/"+name
            content=urllib.request.urlopen(url,timeout=60).read()
            if hashlib.sha256(content).hexdigest()!=expected:raise RuntimeError(f"upstream checksum mismatch: {name}")
            path.write_bytes(content)
        print(name+": pinned source verified")
if __name__=="__main__":main()
