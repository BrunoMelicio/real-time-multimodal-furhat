<div align="center">

# Real-Time Multimodal Interaction with Furhat

> **Disclaimer:** This is a proof of concept developed for a demo using AI tools. A fully tested and manually reviewed final version will be released soon.

**See · Listen · Understand · Respond**

A lightweight Python toolkit for natural, multimodal interaction with individuals and groups.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Furhat](https://img.shields.io/badge/Robot-Virtual_Furhat-7356BF)
![Language model](https://img.shields.io/badge/Language_model-Ollama_%C2%B7_Groq_%C2%B7_OpenAI-222222)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Hardware](https://img.shields.io/badge/Tested_on-Apple_M3_%C2%B7_8_GB-555555)

[🚀 Setup](#setup) · [▶️ Run](#run) · [🧠 Models](#models) · [✨ Capabilities](#capabilities) · [🛠️ Architecture](docs/architecture.md)

</div>

![Two-person interaction: camera, perception view and Virtual Furhat](docs/assets/interaction-demo.png)

*Recorded interaction on an 8 GB MacBook Air: clean camera view on the left, tracked participants and visual cues in the middle, Virtual Furhat on the right.*

<a id="capabilities"></a>

## ✨ What it does

| Capability | Behavior |
|---|---|
| 👀 Visual attention | Detect participants, follow them and direct the robot toward the person being addressed. |
| 🎙️ Active-speaker detection | Combine audio and face video to associate a spoken turn with a tracked participant. |
| 💬 Conversation | Transcribe speech locally and give the language model speech, names and observed visual context. |
| 🥤 Object awareness | Recognize objects and acknowledge the object a newcomer brings into the interaction. |
| ✋ Gestures and games | Detect hand signs, associate hands with participants and score three rock–paper–scissors rounds. |
| 🙂 Responsive behavior | Use head movement, approximate gaze and hand cues; nod while listening and handle experimental interruptions during conversation. |
| 🤖 Embodied responses | Use Furhat's native voice, lip synchronization, head movement and facial gestures. |

**One ongoing interaction.** Start the app, introduce yourself and talk naturally. Furhat receives speech together with observed objects, posture, gaze and gestures. Ask to play a game and it switches into game mode; after scoring three rounds, it invites reflection and returns to conversation. Names and game results stay in session memory—no scenario or winner command is needed.

The continuous app is a **new integration awaiting live validation**. Its individual demonstration components have been recorded successfully. The current release supports individual interaction and a two-participant group; **larger groups are coming soon**. Speech is currently turn based, with experimental interruption handling; simultaneous speech separation and fully conversational full-duplex operation are planned.

<a id="setup"></a>

## 🚀 Get started

### 1 · Prepare the robot and choose a language provider

- Install **Python 3.12** and the **Furhat SDK**. For local language generation, also install **[Ollama](https://ollama.com/)**.
- Start Virtual Furhat and enable its **Realtime API**.
- Select a working voice in the SDK and verify speech and lip synchronization there.
- For the default local provider, download the language model:

```bash
ollama pull llama3.2:1b
```

If Ollama is not already running, keep `ollama serve` running in a separate terminal.

Groq, OpenAI and other OpenAI-compatible Chat Completions services are optional alternatives. They do not require Ollama. See [language provider configuration](#language-providers) below.

### 2 · Clone and install

```bash
git clone https://github.com/BrunoMelicio/real-time-multimodal-furhat.git
cd real-time-multimodal-furhat
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Use this **one environment** for the app and all demos. You do not need to activate it: the commands explicitly select its Python executable.

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

This checks dependencies and perception model files. It does **not** open devices, load models or test provider/robot connections. Use `--check-llm` to test the selected language provider separately.

> **Existing development checkout:** use `../.venv/bin/python` instead of `.venv/bin/python` when the environment is in the parent directory. Keep the existing environment rather than creating another copy.

<a id="run"></a>

## ▶️ Run an interaction

### Individual interaction

```bash
.venv/bin/python main.py --people=1 --mic=builtin
```

### Group interaction

```bash
.venv/bin/python main.py --people=2 --mic=builtin
```

### Larger groups · coming soon 🚧

Support for more than two participants is on the roadmap. For now, `--people` accepts `1` or `2` and sets participant capacity.

**Arrange the windows, then press Space** in the preview to start. **Q / Esc** closes the app. You can begin a group session with one participant; a newcomer is introduced when they join.

Names are normally collected through conversation. Optionally supply them in **left-to-right preview order**:

```bash
.venv/bin/python main.py --people=2 --mic=iphone --names Alice Bob --profile
```

| Option | Use |
|---|---|
| `--people=1` / `--people=2` | Configure individual or group interaction. |
| `--names Alice Bob` | Optional participant names; all named participants must be visible before Space. |
| `--mic=builtin` / `--mic=iphone` | Select a microphone by name; numeric device IDs can change. |
| `--camera=0` | Select the camera. |
| `--host=127.0.0.1` | Connect to local Virtual Furhat; change for another host. |
| `--barge-in=visual` / `vad` | Interrupt Furhat in either people mode. Default `visual` checks participant face/audio activity; `vad` stops on speech onset before speaker identification finishes. |
| `--asr=whistle` / `phonon` / `whisper-tiny` / `whisper-base` / `whisper-small` | Choose local speech recognition. Default: Whisper Base. |
| `--asr-model=MODEL` | Optional model override, e.g. `--asr=whisper --asr-model=tiny.en`. `--whisper=base.en` remains supported. |
| `--object-model=ssd` / `lite0` / `lite2` | Change the object detector. |
| `--llm=ollama` / `groq` / `openai` / `compatible` | Select the language provider; Ollama is the default. `--llm-provider` is an alias. |
| `--llm-model=llama3.2:1b` | Select the provider's model ID. `--model` remains an alias. Groq defaults to `openai/gpt-oss-20b`; OpenAI/compatible require a model. |
| `--llm-base-url=URL` | Override the API base URL; required for another compatible service. |
| `--llm-key-env=VARIABLE_NAME` | Choose the environment variable containing a custom provider key. |
| `--check-llm` | Test model availability and JSON generation without opening devices; cloud checks make an API request. |
| `--profile` | Print frame-stage timings every three seconds. |

### Local speech recognition

Whistle and Phonon are optional local backends. Install their runtimes once (already installed in this development checkout):

```bash
.venv/bin/python -m pip install -r requirements-asr.txt
```

```bash
# Groq for conversation, Whistle on the CPU for speech.
.venv/bin/python main.py --people=1 --mic=iphone --llm=groq --asr=whistle

# Phonon-2 uses MLX on Apple Silicon.
.venv/bin/python main.py --people=1 --mic=iphone --llm=groq --asr=phonon

# Whisper alternatives through whisper.cpp.
.venv/bin/python main.py --people=1 --mic=iphone --llm=groq --asr=whisper-tiny
.venv/bin/python main.py --people=1 --mic=iphone --llm=groq --asr=whisper-base
```

Use one equals sign: `--asr=whistle`. Quotes are optional (`--asr="whistle"` works too). All recognizers receive the same 16 kHz audio and Silero speech boundaries, load once in the ASR worker, and save no audio. New weights download into the runtime's cache on first use. Phonon can take longer to initialize MLX shaders on its first decode. Startup logs name the selected engine; `decode_ms` measures transcription time, while `final_after_voice_ms` also includes the speech-end pause and queueing.

Both one- and two-person modes support interrupting **Furhat**: participant speech stops the current robot utterance, the app listens to the new turn, and the next response can revise the interrupted answer. Human participants are assumed to take turns; overlapping participant speech is not separated. This applies to conversation, invitations, game instructions and post-game follow-ups.

The same `--barge-in` options apply to both people modes. Default `visual` combines Silero VAD with Light-ASD face/audio evidence to reduce interruptions caused by Furhat's own loudspeaker voice. `--barge-in=vad` stops on sustained speech onset without waiting to identify who spoke; group speaker attribution continues while listening. VAD alone also hears robot playback, so use `visual` with loudspeaker echo or VAD-only with headphones/acoustic isolation. While listening, one fresh visible face supplies the speaker identity; multiple faces use Light-ASD. This depends on who is present, not the configured capacity. Interrupted speech retains its onset audio. Empty transcriptions are ignored, including during game rounds.

Hand ownership uses the same geometry in both modes: prefer matching pose wrists, then use a clear participant region if wrist landmarks are unavailable. One person can arrive before the other in `--people=2`; their hands do not need to wait for a second participant. Overlapping regions, competing wrists, duplicate wrist detections and distant hands remain unassigned. The preview labels unnamed owners as `Person 1` / `Person 2` and shows the reason for unassigned hands.

Nods and head shakes answer invitations, game-rule questions and conversational questions through the same handling as spoken yes/no responses. Gestures made during a prompt or its short listening pause are buffered briefly; a spoken response takes precedence. Responses from an addressed participant are kept separate, and old gestures do not answer a later question. Head-motion smoothing uses elapsed time so lower camera frame rates do not continually reset detection. This remains camera-based motion inference, not a guarantee that every human nod will be recognized.

An explicit request to play enters game setup directly; mentioning games can still lead to an invitation. During setup, a collective reply such as “we know the rules” answers the group question. Individual answers prompt the remaining player, and anyone needing an explanation gets the rules before play. A request to start can also proceed with a short explanation. Setup stays with the game controller: unrelated or misrecognized speech cannot turn it into an LLM-improvised countdown. Short replies without a speaker match can answer a pending group game question without assigning them to a guessed person; names and personal answers still require identity. Terminal `[STATE]` entries distinguish setup, hand release and active play.

<a id="language-providers"></a>

### Language providers and API keys

The flags separate **who serves the model** (`--llm`) from **which model** (`--llm-model`). Existing commands without these flags keep using local Ollama.

**Groq:** copy `.env.example` to `.env` in this repository, then edit the key there:

```dotenv
GROQ_API_KEY=paste_your_key_here
```

The app automatically reads this checkout's `.env` regardless of the current working directory. `.env` is ignored by Git; exported shell variables take precedence. Never put an actual key in `.env.example` or command-line flags.

```bash
# Test the key/model without the camera, microphone or Furhat.
.venv/bin/python main.py --llm=groq --llm-model=openai/gpt-oss-20b --check-llm

# Run the interaction with Groq.
.venv/bin/python main.py --people=1 --mic=iphone --llm=groq --llm-model=openai/gpt-oss-20b

# Run with local Ollama instead.
.venv/bin/python main.py --people=1 --mic=iphone --llm=ollama --llm-model=llama3.2:1b
```

To make Groq the default, also put `LLM_PROVIDER=groq` and `LLM_MODEL=openai/gpt-oss-20b` in `.env`. CLI flags override those defaults.

For **OpenAI**, set `OPENAI_API_KEY` in `.env` and use `--llm=openai --llm-model=YOUR_MODEL_ID`. For **another compatible service**, set `LLM_API_KEY` and use:

```bash
.venv/bin/python main.py --llm=compatible --llm-base-url=https://your-provider.example/v1 --llm-model=YOUR_MODEL_ID --check-llm
```

The compatible service must support Chat Completions with JSON object responses. You can use `--llm-key-env=YOUR_PROVIDER_API_KEY` to read a differently named key. `LLM_BASE_URL` and `LLM_API_KEY_ENV` are the corresponding optional environment settings. `--ollama-url` remains a local-only alias for an Ollama endpoint.

**Cloud data flow:** transcription, perception and speaker detection stay on your Mac. The selected cloud language provider receives the recognized speech, names, recent conversation, summarized visual cues and game results. Camera frames and raw microphone audio are not included in those language requests. Speech output still uses Furhat's selected voice.

### 🎮 What interaction can look like

- **Welcome:** Furhat notices a participant, asks their name and keeps it for the session. A newcomer is welcomed after the current spoken turn.
- **Object-aware conversation:** hold a bottle/cup clearly in view; ask about it or let the detected object inform the reply.
- **Play:** after four chat turns, or earlier if you mention play, games or fun, the next model reply naturally bridges into an invitation to play rock–paper–scissors. Say yes to enter the game, or decline and continue chatting. One free-form model call generates each conversational reply; there is no separate intent classifier or evidence-quote requirement. Python handles the game rules, observed hand signs and scoring.
- **Gesture answers:** respond to invitations, rules and conversational questions with speech, a nod or a head shake. Show a fist, palm or V sign for rock, paper or scissors.
- **Reflect:** game results automatically become conversation context. Furhat asks how the game felt and can acknowledge observed downward posture or facial movement.
- **Interrupt:** experimental speech/near-face-motion handling can stop a robot reply and continue with the participant's new turn. Visual cues are uncertain; safely mime near-face movement without contact if demonstrating it.

The app loads the selected perception components once and keeps them active. It saves no recordings or run files. Small standalone demonstrations remain under `demos/` for component checks.

<a id="models"></a>

## 🧠 Models and hardware

| Component | Selected implementation | Role |
|---|---|---|
| Person detection/tracking | [YOLO26 Nano](https://github.com/ultralytics/ultralytics) | Maintain participant boxes and short-lived track IDs. |
| Face/head/gaze | [MediaPipe Face Landmarker](https://github.com/google-ai-edge/mediapipe) + geometry | Estimate head orientation and relative iris displacement. |
| Body | [MediaPipe Pose Landmarker Lite](https://github.com/google-ai-edge/mediapipe) | Body landmarks and wrist evidence for hand ownership. |
| Hands | [MediaPipe Gesture Recognizer / Hand Landmarker](https://github.com/google-ai-edge/mediapipe) | Game signs and hand movement near the face. |
| Objects | [SSD MobileNetV2](https://github.com/tensorflow/models/tree/master/research/object_detection) or [EfficientDet-Lite0/Lite2](https://github.com/google/automl/tree/master/efficientdet) | Lightweight object recognition; the continuous app defaults to Lite0. |
| Speech activity | [Silero VAD](https://github.com/snakers4/silero-vad) | Detect speech boundaries and support interruptions. |
| Active speaker | [Light-ASD](https://github.com/Junhua-Liao/Light-ASD), approximately 1.02M parameters | Identify group speakers and provide face/audio interruption evidence in either people mode with the default visual echo guard. |
| Transcription | [Whisper](https://github.com/openai/whisper) via [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [Whistle](https://cactuscompute.com/blog/whistle), or [Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) | Local selectable ASR; Base/Tiny/Small, Whistle CPU, or Phonon MLX. |
| Language | [Ollama](https://github.com/ollama/ollama) / [Llama 3.2 1B](https://github.com/meta-llama/llama-models), or cloud [Groq](https://console.groq.com/docs/openai) / OpenAI / compatible services | Context-aware responses and semantic game intent; local Ollama remains the default. |
| Voice/lip sync | [Furhat native speech](https://docs.furhat.io/realtime-api/events) | Retain the SDK's synchronized facial animation. |

**Development hardware:** MacBook Air **M3, 8 GB unified memory**, webcam, built-in or iPhone microphone, and **Virtual Furhat SDK 2.9.3**. OBS was used to record demonstrations; it is not required to run them.

The continuous app runs MediaPipe, YOLO, Light-ASD, Whistle and the default Whisper backend on **CPU**. Phonon uses MLX on Apple Silicon. Ollama manages its own acceleration; it is not used when a cloud provider is selected. The smaller local language fallback is [Gemma 3 270M](https://github.com/google/gemma_pytorch). **Built with Llama** when using the default language model.

### 📊 Validation and practical limits

- The separate individual and group demonstrations were run and recorded by the project author.
- A recent two-person reflection log on the M3 showed approximately **13–18 FPS** with the required perception stack active. This is a historical standalone-demo run, not a measurement of the new continuous app or a guaranteed rate.
- Offline regression tests check scene transitions, gesture policies, speaker ownership, cue handling and shutdown with synthetic data/mocks.
- Fresh-machine installation of the pinned dependencies still needs validation. MediaPipe and Ultralytics request different OpenCV distributions; avoid uninstalling either from a working environment in place.

Tracking is spatial, **not biometric identity**. Iris-based gaze is approximate. Observable posture/hand movement does not establish emotions, injury or clinical effectiveness. The continuous individual game commits random robot moves before reading hand signs; the group referee scores detected signs.

## 🧪 Run the tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests do not open a camera/microphone, start the robot or run model inference. Environments, model weights, recordings, participant transcripts and local rollback notes are excluded from Git.

## 📚 Learn more

- [🛠️ Architecture and execution flow](docs/architecture.md)
- [📜 Third-party source, models and licenses](docs/third_party.md)

## 📜 License

The project's **original source code is licensed under [MIT](LICENSE)**, allowing use, modification and redistribution under its terms.

**Check the license of every dependency and model checkpoint before use, redistribution or deployment.** The MIT license here does not relicense third-party code, weights or SDK services. In particular, [Ultralytics YOLO uses AGPL-3.0 or an Enterprise license](https://www.ultralytics.com/license), and [Llama 3.2 has its own Community License and use policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE). Their obligations can apply to a combined application; this repository's MIT label does not remove them.

## ✅ To do

- [ ] Support groups with **more than two participants**.
- [ ] Create a **Hugging Face Spaces demo**.
- [ ] Integrate with **other robots** through replaceable robot adapters.
- [ ] Add **full-duplex, non-turn-based conversation**, with continuous listening, natural interruptions and overlapping speech handling.
- [ ] Validate the continuous app on additional hardware and expand end-to-end tests.
