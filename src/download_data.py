"""Fetch upstream example data and verify the exact analyzed version."""
from pathlib import Path
import argparse,hashlib,urllib.request
ROOT=Path(__file__).resolve().parents[1]
URL='https://raw.githubusercontent.com/google/meridian/refs/heads/main/meridian/data/simulated_data/csv/geo_all_channels.csv'
SHA256='d9ee016f7cd21f5c90b50da10a794af91a372b69f881d3caa266edd41fdafbf6'
p=argparse.ArgumentParser();p.add_argument('--local-file',type=Path);args=p.parse_args()
content=args.local_file.read_bytes() if args.local_file else urllib.request.urlopen(URL,timeout=60).read()
actual=hashlib.sha256(content).hexdigest()
if actual!=SHA256:raise SystemExit(f'Dataset version mismatch: {actual}. Upstream may have changed; do not silently reproduce against another version. Supply the documented version with --local-file.')
out=ROOT/'data/geo_all_channels.csv';out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(content)
print(f'Dataset verified: {actual}')
