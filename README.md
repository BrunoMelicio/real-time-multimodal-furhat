<div align="center">

# Real-Time Multimodal Interaction with Furhat

**See · Listen · Understand · Respond**

A lightweight Python toolkit for embodied interaction with one or two people.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Furhat](https://img.shields.io/badge/Robot-Virtual_Furhat-7356BF)
![Local AI](https://img.shields.io/badge/Language_model-Local_Ollama-222222)
![Hardware](https://img.shields.io/badge/Tested_on-Apple_M3_%C2%B7_8_GB-555555)

[🚀 Setup](#setup) · [▶️ Run](#run) · [🧠 Models](#models) · [🎬 Scenarios](docs/scenarios.md) · [🛠️ Architecture](docs/architecture.md)

</div>

![Two-person interaction: camera, perception view and Virtual Furhat](docs/assets/interaction-demo.png)

*Recorded interaction on an 8 GB MacBook Air: clean camera view on the left, tracked participants and visual cues in the middle, Virtual Furhat on the right.*

## ✨ What it does

| Capability | Behavior |
|---|---|
| 👀 Visual attention | Detect participants, follow them and direct the robot toward the person being addressed. |
| 🎙️ Active-speaker detection | Combine audio and face video to associate a spoken turn with a tracked participant. |
| 💬 Conversation | Transcribe speech locally and give the language model speech, names and observed visual context. |
| 🥤 Object awareness | Recognize objects and acknowledge the object a newcomer brings into the interaction. |
| ✋ Gestures and games | Detect hand signs, associate hands with participants and score three rock–paper–scissors rounds. |
| 🙂 Responsive behavior | Use head movement, approximate gaze and hand cues; nod while listening and handle interruptions in the reflection scenario. |
| 🤖 Embodied responses | Use Furhat's native voice, lip synchronization, head movement and facial gestures. |

**Three scenarios, one entrypoint.** Choose introduction, game or reflection; each runs independently. A continuous session joining all three scenarios is planned. The current two-person scenes expect people to speak one at a time; speech separation is outside this release.

<a id="setup"></a>

## 🚀 Get started

### 1 · Prepare the robot and local language model

- Install **Python 3.12**, the **Furhat SDK** and **[Ollama](https://ollama.com/)**.
- Start Virtual Furhat and enable its **Realtime API**.
- Select a working voice in the SDK and verify speech and lip synchronization there.
- Download the local language model:

```bash
ollama pull llama3.2:1b
```

If Ollama is not already running, keep `ollama serve` running in a separate terminal.

### 2 · Clone and install

```bash
git clone https://github.com/BrunoMelicio/real-time-multimodal-furhat.git
cd real-time-multimodal-furhat
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use this **one environment** for every scenario. You do not need to activate it: the commands explicitly select its Python executable.

### 3 · Get the perception models

```bash
.venv/bin/python tools/setup_models.py --download
```

This downloads the selected small perception checkpoints and verifies their hashes against [the model manifest](models/manifest.json). Weights are kept locally and excluded from Git. Whisper.cpp may download `base.en` to its own cache on first use.

Already have the perception weights? Copy and verify them instead:

```bash
.venv/bin/python tools/setup_models.py --from-directory /path/to/existing/models
```

### 4 · Check the installation

```bash
.venv/bin/python main.py --check
```

This checks dependencies and perception model files. It does **not** open devices, load models or test connections to Furhat/Ollama.

> **Existing development checkout:** use `../.venv/bin/python` instead of `.venv/bin/python` when the environment is in the parent directory. Keep the existing environment rather than creating another copy.

<a id="run"></a>

## ▶️ Run an interaction

### One person

```bash
# Arrival, introduction and object-aware conversation
.venv/bin/python main.py --people=1 --scenario=introduction --mic=builtin --object-model=lite0

# Play three rounds against Furhat
.venv/bin/python main.py --people=1 --scenario=game --mic=builtin --name=Alice

# Reflect on the game, with visual cues and interruptions
.venv/bin/python main.py --people=1 --scenario=reflection --mic=builtin --name=Alice
```

### Two people

```bash
# Welcome the first participant, then a newcomer holding an object
.venv/bin/python main.py --people=2 --scenario=introduction --mic=builtin --object-model=lite0

# Furhat judges a three-round game between the participants
.venv/bin/python main.py --people=2 --scenario=game --mic=builtin --names Alice Bob

# Discuss the result; use the actual winner's name
.venv/bin/python main.py --people=2 --scenario=reflection --mic=builtin --names Alice Bob --winner Bob --profile
```

**Before starting:** arrange the camera, perception and Virtual Furhat windows. Then press **Space** in the preview to begin. Press **Q** or **Esc** in the preview to close.

| Option | Use |
|---|---|
| `--mic=builtin` / `--mic=iphone` | Select a microphone by name; numeric device IDs can change between runs. |
| `--camera=0` | Select the camera. |
| `--host=127.0.0.1` | Connect to local Virtual Furhat; change for another host. |
| `--whisper=base.en` / `--whisper=small.en` | Prefer Base for speed; try Small for accuracy. |
| `--object-model=ssd` / `lite0` / `lite2` | Change the object detector in introductions. |
| `--names Alice Bob` | Two-person game/reflection: names **left to right in the preview**. |
| `--winner Bob` | Two-person reflection: pass the real game winner. |
| `--profile` | Two-person scenes: print frame-stage timings every three seconds. |

Introduction/reflection also accept `--model=gemma3:270m` as a smaller Ollama fallback, if installed. The game dialogue is scripted. See the [scenario guide](docs/scenarios.md) for participant lines, gestures and expected events.

<a id="models"></a>

## 🧠 Models and hardware

| Component | Selected implementation | Role |
|---|---|---|
| Person detection/tracking | YOLO26 Nano in two-person scenes | Maintain participant boxes and short-lived track IDs. |
| Face/head/gaze | MediaPipe Face Landmarker + geometry | Estimate head orientation and relative iris displacement. |
| Body | MediaPipe Pose Landmarker Lite | Body landmarks and wrist evidence for hand ownership. |
| Hands | MediaPipe Gesture Recognizer / Hand Landmarker | Game signs and hand movement near the face. |
| Objects | SSD MobileNetV2 or EfficientDet-Lite0/Lite2 | Lightweight object recognition; default is SSD. |
| Speech activity | Silero VAD | Detect speech boundaries and support interruptions. |
| Active speaker | Light-ASD, approximately 1.02M parameters | Match audio activity to visible face video. |
| Transcription | whisper.cpp `base.en` via pywhispercpp | Local ASR; `small.en` is optional. |
| Language | Ollama `llama3.2:1b` | Context-aware responses; Gemma 270M is the smaller fallback. |
| Voice/lip sync | Furhat native speech | Retain the SDK's synchronized facial animation. |

**Development hardware:** MacBook Air **M3, 8 GB unified memory**, webcam, built-in or iPhone microphone, and **Virtual Furhat SDK 2.9.3**. OBS was used to record demonstrations; it is not required to run them.

MediaPipe, Light-ASD and the default Whisper backend run on **CPU**. Two-person YOLO also uses CPU. The single-person optional YOLO object path can use MPS; Ollama manages its own acceleration. These are selected runtime choices, not a guarantee that GPU execution is faster.

### 📊 Validation and practical limits

- The separate single-person and two-person scenarios were run and recorded by the project author.
- A recent two-person reflection log on the M3 showed approximately **13–18 FPS** with the required perception stack active. This is an observed run, not a controlled benchmark or a guaranteed rate.
- Offline regression tests check scene transitions, gesture policies, speaker ownership, cue handling and shutdown with synthetic data/mocks.
- Fresh-machine installation of the pinned dependencies still needs validation. MediaPipe and Ultralytics request different OpenCV distributions; avoid uninstalling either from a working environment in place.

Tracking is spatial, **not biometric identity**. Iris-based gaze is approximate. Observed posture/hand movement does not establish emotions, injury or clinical effectiveness. The single-person game uses a rehearsed, precommitted robot move sequence; the two-person referee scores detected signs. [More details →](docs/scenarios.md)

## 🗂️ Repository map

```text
main.py                     # Common launcher
requirements.txt            # One environment
furhat_interaction/         # All runtime code and shared components
  single_person/            # Introduction, game, reflection
  multi_person/             # Two-person scenes and runtime
  audio/                    # Voice activity detection
  group/                    # Robot control, hand ownership, referee
  third_party/              # Licensed Light-ASD architecture
demos/                     # Small runnable entrypoints
  single_person/            # One-person scenarios
  multi_person/             # Two-person scenarios
  components/               # Robot connection and speech checks
docs/                      # Scenarios, architecture, credits and visuals
models/manifest.json        # Checkpoint URLs, sizes and hashes
tools/setup_models.py       # Download/copy and verify checkpoints
tests/                     # Offline regression checks
```

## 🧪 Run the tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests do not open a camera/microphone, start the robot or run model inference. Environments, model weights, recordings, participant transcripts and local rollback notes are excluded from Git.

## 📚 Learn more

- [🎬 Scenario scripts and controls](docs/scenarios.md)
- [🛠️ Architecture and execution flow](docs/architecture.md)
- [📜 Third-party source and model notices](docs/third_party.md)

Furhat, MediaPipe, Ultralytics, Light-ASD, Whisper and Ollama remain subject to their upstream terms. This repository has not yet selected a license for its original code; public visibility alone is not a license grant.
