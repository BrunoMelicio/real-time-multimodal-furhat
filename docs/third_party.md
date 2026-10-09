# Third-party source and model provenance

Only Light-ASD architecture source is vendored, from nvthach124/Light-ASD commit `9d2a6db1bf27929dfff5a460438598130bc7bd8e`. Its MIT license is retained in `speaker_models/light/LICENSE`. Changes from upstream are relative imports, not altered architecture dimensions. The curated adapter uses the tested strict checkpoint-loading path and omits unused TalkNet/Dolphin branches.

Dependencies and pretrained models retain their respective upstream licenses; installing a Python package does not change its license or grant rights to its model weights. MediaPipe task sources and official URLs, checkpoint sizes and tested SHA-256 values are in `models/manifest.json`. YOLO26n is obtained through Ultralytics. Furhat SDK and Ollama/Llama are external prerequisites.

No original-project license grant is selected in this initial private repository. Check the complete dependency/model license terms before choosing a publication license or making the repository public. No third-party model weights, SDK distribution, or private participant data are bundled.
