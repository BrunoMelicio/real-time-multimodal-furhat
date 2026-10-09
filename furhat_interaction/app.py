"""Select a tested scene without importing models or opening hardware early."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys

from furhat_interaction.paths import ROOT


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
    parser=argparse.ArgumentParser(description='Real-Time Multimodal Interaction with Furhat: interaction scene launcher.')
    parser.add_argument('--people',type=int,choices=(1,2),default=2)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--scenario', choices=('introduction','game','reflection'),
                           help='Choose an interaction scenario (default: introduction).')
    selection.add_argument('--part',type=int,choices=(1,2,3),help=argparse.SUPPRESS)
    parser.add_argument('--check',action='store_true',help='Check local dependencies/weights without using hardware.')
    args,remaining=parser.parse_known_args(argv)
    if args.check:
        if remaining: parser.error('Unexpected arguments with --check: '+' '.join(remaining))
        return check()
    scenarios=('introduction','game','reflection')
    part=args.part or (scenarios.index(args.scenario)+1 if args.scenario else 1)
    scenario=scenarios[part-1]
    if args.people==2:
        from furhat_interaction.multi_person.runtime import run
        return run(part,remaining)
    module=importlib.import_module(f'furhat_interaction.single_person.{scenario}')
    # Existing single-person mains use argparse's normal process arguments.
    previous=sys.argv
    try:
        sys.argv=[str(ROOT/'furhat_interaction'/'single_person'/f'{scenario}.py'),*remaining]
        result=module.main()
        return result if isinstance(result,int) else 0
    finally:
        sys.argv=previous
