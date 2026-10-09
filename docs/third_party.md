# 📜 Code, model provenance and licenses

The project's original code is provided under [MIT](../LICENSE). **This does not grant a new license to third-party source, model weights, SDK software or services.** Check the terms for each exact version/checkpoint and intended use before redistributing or deploying the application.

## Upstream terms

| Component | Source / license reference |
|---|---|
| MediaPipe framework | [Official repository and Apache-2.0 license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE). Review the specific task/model assets separately. |
| SSD / EfficientDet | [TensorFlow Models](https://github.com/tensorflow/models/tree/master/research/object_detection), [Google EfficientDet](https://github.com/google/automl/tree/master/efficientdet). Verify each checkpoint's terms, not just the framework license. |
| Ultralytics YOLO code and models | [AGPL-3.0 / Enterprise licensing](https://www.ultralytics.com/license). These terms can impose obligations on the combined application; the MIT license on our original files does not waive them. |
| Light-ASD | [Official source](https://github.com/Junhua-Liao/Light-ASD), [MIT license](https://github.com/Junhua-Liao/Light-ASD/blob/main/LICENSE). |
| Silero VAD | [Official source and MIT license](https://github.com/snakers4/silero-vad/blob/master/LICENSE). |
| Whisper | [Official source](https://github.com/openai/whisper). Check the selected model and source terms. |
| whisper.cpp / pywhispercpp | [whisper.cpp](https://github.com/ggml-org/whisper.cpp), [Python bindings](https://github.com/absadiki/pywhispercpp). Retain upstream notices when redistributing. |
| Llama 3.2 | [Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE), including its linked acceptable-use policy. **Built with Llama** when the default model is used. |
| Gemma fallback | [Official implementation](https://github.com/google/gemma_pytorch); review the selected Gemma model's terms. |
| Ollama | [Official source](https://github.com/ollama/ollama). Its software license does not replace licenses of models it serves. |
| Furhat | [Official developer documentation](https://docs.furhat.io/). SDK, voices and services retain Furhat/provider terms. |

Check transitive dependencies too. A software package's license and its pretrained model's license may differ. Downloading weights separately does not remove upstream obligations. This notice is a reference, not a guarantee that a particular deployment satisfies every applicable term.

## Vendored Light-ASD architecture

The architecture was obtained from [nvthach124/Light-ASD](https://github.com/nvthach124/Light-ASD), commit `9d2a6db1bf27929dfff5a460438598130bc7bd8e`, a derivative of the official repository above. Its [MIT license is retained locally](../furhat_interaction/third_party/light/LICENSE). The source adapts relative imports without changing architecture dimensions. The adapter uses the tested strict-loading path for the TalkSet checkpoint.

## Assets and checkpoints

Official checkpoint URLs, sizes and tested SHA-256 values are listed in [the model manifest](../models/manifest.json). Weights, the Furhat SDK and participant recordings/transcripts are excluded from Git. The README screenshot was supplied by the project author for publication.
