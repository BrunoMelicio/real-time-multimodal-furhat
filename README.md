# Real-Time Multimodal Interaction with Furhat

A Python research prototype combining visual attention, active-speaker detection, speech recognition, local language generation, gestures and game interaction with Virtual Furhat.

## Current release

This initial repository preserves the recorded single-person and two-person conference scenes and their shared components. `main.py` selects an individual scene. **It is not yet the unified continuous application:** loading the full perception stack once and carrying an in-memory session across all three scenes is the next milestone. No automatic hardware test or model download was performed during repository preparation.

- Part 1: introductions, attention to speakers/newcomers and object conversation.
- Part 2: spoken/head-gesture rules answers and three scored rock–paper–scissors rounds.
- Part 3: game reflection, observed downward posture, listening acknowledgment and interruptions during encouragement.

Two-person interaction uses tracked spatial seats, not biometric identity; speaker attribution uses evidence captured during speech. Gaze is an iris-based approximation. Movement cues do not establish emotions, diagnoses, injury or clinical effectiveness. Separation is outside this initial core release.

## Setup: one environment

Use Python 3.12, a webcam/microphone, Virtual Furhat with Realtime API enabled and Ollama with `llama3.2:1b`. Select a Furhat voice that supports lip synchronization. First verify native speech works in the SDK.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/setup_models.py --download
ollama pull llama3.2:1b
```

If Ollama is not already running, run `ollama serve` in another terminal. Model setup verifies the recorded checkpoint hashes and fails if upstream assets have changed. Download/install commands are user-run; weights are excluded from Git.

For existing local weights instead of downloads:

```bash
.venv/bin/python tools/setup_models.py --from-directory /path/to/existing/models
```

Whisper.cpp's first recognition run may download `base.en` into its own cache. Current defaults: MediaPipe CPU, YOLO CPU, Light-ASD CPU, Whisper.cpp Base CPU and Ollama Llama1B. Fresh installation of the pinned direct dependencies on a second machine has not yet been validated. MediaPipe and Ultralytics declare different OpenCV distributions; do not uninstall either from a working environment in place.

## Run

```bash
.venv/bin/python main.py --check
.venv/bin/python main.py --people=2 --part=1 --mic=builtin --object-model=lite0
.venv/bin/python main.py --people=2 --part=2 --mic=builtin --names Bruno Maria
.venv/bin/python main.py --people=2 --part=3 --mic=builtin --names Bruno Maria --winner Maria --profile
```

Names in the two-person scenes are left-to-right in the mirrored preview. Supply the actual Part 2 winner. Parts 2/3 can read older seat/session files for compatibility, but two-person runs do not save new files. Space starts; Q/Esc closes. Use `--mic=iphone` for an available iPhone microphone; numeric IDs can change. The recorded single-person scripts use `--people=1`; their existing arguments, session handoff and save behavior are retained. See [rehearsal](docs/rehearsal.md).

On the original development laptop, use the existing parent environment—do not create a second one:

```bash
../.venv/bin/python main.py --check
../.venv/bin/python main.py --people=2 --part=1 --mic=iphone --object-model=lite0
```

## Demos and tests

`demos/conference/` contains six scene entrypoints. `demos/components/` contains connection and speech/expression checks; additional selected component demos will be migrated incrementally. Helpers extracted during curation avoid importing entire speech-separation benchmarks merely to select a microphone or clean text.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

These tests use synthetic inputs and fake device owners, not a live robot. Model weights, environments, recordings, participant transcripts, caches and local rollback notes are not included. See [third-party notices](docs/third_party.md) and [architecture](docs/architecture.md).
