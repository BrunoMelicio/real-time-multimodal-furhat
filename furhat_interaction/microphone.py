"""Microphone selection, extracted from the tested audio helpers."""
def microphone_argument(value):
    try:
        return int(value)
    except ValueError:
        return value

def choose_microphone(sd, requested=None):
    """Automatic room input prefers a laptop mic; explicit choices always win."""
    catalog = sd.query_devices()
    inputs = [dict(info,index=info.get('index',index)) for index,info in enumerate(catalog)
              if info.get('max_input_channels',0)>0]
    if not inputs:
        raise RuntimeError('No input devices found. Check microphone connection and permission.')
    builtins = []
    for info in inputs:
        name = info['name'].casefold()
        if ('macbook' in name or 'imac microphone' in name or
            any(word in name for word in ('built-in microphone','built in microphone','internal microphone','internal mic'))):
            builtins.append(info)
    automatic = requested is None or (isinstance(requested,str) and requested.casefold()=='auto')
    builtin = isinstance(requested,str) and requested.casefold()=='builtin'
    system = isinstance(requested,str) and requested.casefold()=='default'
    if (automatic or builtin) and builtins:
        return builtins[0]['index'],builtins[0],inputs,'built-in microphone for room speech'
    if builtin:
        names = ', '.join(f'{info["index"]}: {info["name"]}' for info in inputs)
        raise RuntimeError('Built-in microphone not identified. Choose --mic=INDEX or a name. Inputs: '+names)
    if automatic or system:
        info = sd.query_devices(kind='input')
        return info['index'],info,inputs,'system default input'
    info = sd.query_devices(requested,kind='input')
    return info['index'],info,inputs,'explicit microphone choice'
def show_inputs(sd):
    print('[MIC INPUTS] Current inputs for this run:', flush=True)
    for index, info in enumerate(sd.query_devices()):
        if info['max_input_channels'] > 0:
            print(f'  ID {info.get("index", index)}: {info["name"]}', flush=True)

def select_microphone(sd, requested):
    show_inputs(sd)
    try:
        device, info, _, reason = choose_microphone(sd, requested)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(f'Cannot select microphone {requested!r}: {exc}. '
                           'IDs from earlier runs may have changed. Use --mic=builtin '
                           'or --mic="iPhone" for a connected iPhone microphone.') from exc
    print(f'[MIC] Selected {info["name"]} (current ID {device}; {reason})', flush=True)
    if isinstance(requested, int):
        print('[MIC] Numeric IDs are temporary. Prefer --mic=builtin or a microphone name.', flush=True)
    if 'microsoft teams audio' in info['name'].casefold():
        raise RuntimeError('Selected Microsoft Teams Audio, a virtual input rather than a physical microphone. '
                           'Choose a physical microphone with --mic=builtin or --mic="iPhone".')
    return device, info
