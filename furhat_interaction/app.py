"""Select a tested scene without importing models or opening hardware early."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def check():
    modules=('numpy','cv2','mediapipe','torch','ultralytics','sounddevice',
             'faster_whisper','pywhispercpp','furhat_realtime_api','python_speech_features')
    missing=[name for name in modules if importlib.util.find_spec(name) is None]
    manifest=json.loads((ROOT/'models/manifest.json').read_text())
    absent=[item['path'] for item in manifest if not (ROOT/'models'/item['path']).is_file()]
    print('Missing Python modules: '+(', '.join(missing) or 'none'))
    print('Missing model files: '+(', '.join(absent) or 'none'))
    print('This check does not open a microphone/camera, load models, or contact Furhat/Ollama.')
    return int(bool(missing or absent))


def main(argv=None):
    parser=argparse.ArgumentParser(description='Real-Time Multimodal Interaction with Furhat: conference scene launcher.')
    parser.add_argument('--people',type=int,choices=(1,2),default=2)
    parser.add_argument('--part',type=int,choices=(1,2,3),default=1,
                        help='1: introduction/objects; 2: game; 3: reflection/interruptions')
    parser.add_argument('--check',action='store_true',help='Check local dependencies/weights without using hardware.')
    args,remaining=parser.parse_known_args(argv)
    if args.check:
        if remaining: parser.error('Unexpected arguments with --check: '+' '.join(remaining))
        return check()
    if args.people==2:
        from conference_ready_multi.runtime import run
        return run(args.part,remaining)
    module=importlib.import_module(f'conference_ready_{args.part}')
    # Existing single-person mains use argparse's normal process arguments.
    previous=sys.argv
    try:
        sys.argv=[str(ROOT/f'conference_ready_{args.part}.py'),*remaining]
        result=module.main()
        return result if isinstance(result,int) else 0
    finally:
        sys.argv=previous
