"""Explicit user-run model setup; downloaded files must match the manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-directory',type=Path,help='Copy an existing models directory instead of downloading.')
    parser.add_argument('--download',action='store_true',help='Explicitly allow fetching model weights.')
    args=parser.parse_args()
    if bool(args.from_directory)==bool(args.download):
        parser.error('Choose exactly one of --from-directory PATH or --download.')
    manifest=json.loads((ROOT/'models/manifest.json').read_text())
    for item in manifest:
        target=ROOT/'models'/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError(f'Existing model differs from manifest: {item["path"]}; not overwritten.')
            print('[CACHED]',item['path']);continue
        with tempfile.TemporaryDirectory() as tmp:
            temporary=Path(tmp)/Path(item['path']).name
            if args.from_directory:
                source=args.from_directory/item['path']
                if not source.is_file(): raise FileNotFoundError(source)
                shutil.copyfile(source,temporary)
            elif item['url']:
                print('[DOWNLOAD]',item['path'],flush=True)
                with urlopen(item['url'],timeout=120) as response,temporary.open('wb') as out:
                    shutil.copyfileobj(response,out)
            else:
                from ultralytics.utils.downloads import attempt_download_asset
                attempt_download_asset(str(temporary))
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError(f'Checkpoint changed: {item["path"]}. Review the model manifest; no unchecked file retained.')
            # target.parent can be on another filesystem; no atomic rename across volumes.
            shutil.copyfile(temporary,target)
        print('[READY]',item['path'])
    print('Vision/Light-ASD models ready. Whisper.cpp Base and Ollama Llama1B use their own local caches.')


if __name__=='__main__': main()
