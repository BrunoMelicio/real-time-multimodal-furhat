# Architecture and next milestone

Camera -> person/face/body/hand/object perception -> time-stamped cues and tracked participants.
Microphone -> Silero VAD -> Light-ASD speaker association -> Whisper.cpp transcription.
Scene state + actual speech + visual context -> scripted game logic or local Ollama response -> Furhat native speech/head/gesture commands.

Current two-person runtime separates camera capture, audio analysis, transcription and dialogue ownership. A latest-frame camera avoids stale-frame backlogs; hand ownership uses body wrists and rejects ambiguous assignments. Three independent scenes reduce integration risk. All vision required by each scene stays active during that scene.

Next: a new unified session with one model-loading lifecycle, all selected visual components active from startup, configurable camera/detector input size, in-memory names/game results and transitions between the three scenarios. Preserve the separate conference scenes as regression references. Validate single-person operation first, then two-person association. No promise of a particular FPS on an 8 GB laptop; report stage timings and latency rather than only displayed FPS.
