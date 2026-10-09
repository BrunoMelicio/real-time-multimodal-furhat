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
    parser=argparse.ArgumentParser(description='Real-Time Multimodal Interaction with Furhat.',add_help=False)
    parser.add_argument('--people',type=int,choices=(1,2),default=2,
                        help='Participant capacity; more than two coming soon.')
    parser.add_argument('--check',action='store_true',help='Check dependencies/weights without hardware.')
    args,remaining=parser.parse_known_args(argv)
    if args.check:
        if remaining: parser.error('Unexpected arguments with --check: '+' '.join(remaining))
        return check()
    from furhat_interaction.session.runtime import run
    return run(args.people,remaining)


def run_demo(people,scenario,argv=None):
    """Independent demonstrations; the main app does not select scenarios."""
    remaining=list(argv or [])
    part=('introduction','game','reflection').index(scenario)+1
    if people==2:
        from furhat_interaction.multi_person.runtime import run
        return run(part,remaining)
    module=importlib.import_module('furhat_interaction.single_person.'+scenario)
    previous=sys.argv
    try:
        sys.argv=[str(ROOT/'furhat_interaction'/'single_person'/f'{scenario}.py'),*remaining]
        result=module.main()
        return result if isinstance(result,int) else 0
    finally:
        sys.argv=previous
