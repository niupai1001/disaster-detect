"""Acquire only the four user-provided SRF resources; never downloads imagery."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import runpy
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    original = Path('C:/Users/yzh/Downloads/SRF/download_srf_sources.py')
    sources = runpy.run_path(str(original))['SOURCES']
    out = Path(__file__).resolve().parents[2]/'data/physics/srf'
    if not args.download:
        print(json.dumps(sources, ensure_ascii=False, indent=2)); return
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(stage='pre_G1_acquisition_only', original_script_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    per_file_cap_bytes=64*1024**2, total_cap_bytes=256*1024**2,
                    training_allowed=False, scientific_ingestion_performed=False, resources=[])
    received = 0
    for source in sources:
        row = dict(source, retrieved_utc=datetime.now(timezone.utc).isoformat())
        target = out/source['file']; temp = target.with_name(target.name+'.part')
        try:
            if not target.exists():
                req = urllib.request.Request(source['url'], headers={'User-Agent':'Scientific-SRF-Resource-Audit/1.0'})
                with urllib.request.urlopen(req, timeout=45) as response, temp.open('wb') as handle:
                    row['final_url'] = response.url; row['content_type'] = response.headers.get('Content-Type')
                    count = 0
                    while block := response.read(1024*1024):
                        count += len(block); received += len(block)
                        if count>manifest['per_file_cap_bytes'] or received>manifest['total_cap_bytes']:
                            raise RuntimeError('SRF-only transfer cap reached')
                        handle.write(block)
                candidate = temp
            else:
                candidate = target
            # Validate using the final format; .xlsx.part is still an XLSX.
            if target.suffix=='.xlsx':
                with zipfile.ZipFile(candidate) as archive:
                    if 'xl/workbook.xml' not in archive.namelist(): raise ValueError('not an XLSX workbook')
            else:
                with gzip.open(candidate,'rb') as stream:
                    if len(stream.read(512))!=512: raise ValueError('truncated gzip archive')
            if candidate==temp: temp.replace(target); row['status']='downloaded'
            else: row['status']='reused_local'
            row.update(path=str(target), bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        except Exception as exc:
            row.update(status='failed', error=str(exc))
        manifest['resources'].append(row); manifest['received_bytes']=received
        (out/'acquisition_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(id=row['id'],status=row['status'],bytes=row.get('bytes'),error=row.get('error')),ensure_ascii=False),flush=True)
    if any(r['status']=='failed' for r in manifest['resources']): raise SystemExit(2)


if __name__=='__main__': main()
