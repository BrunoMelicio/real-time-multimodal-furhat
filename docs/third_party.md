# Third-party source and model provenance

Only Light-ASD architecture source is vendored, from nvthach124/Light-ASD commit `9d2a6db1bf27929dfff5a460438598130bc7bd8e`. Its MIT license is retained in [the retained MIT license](../furhat_interaction/third_party/light/LICENSE). Changes from upstream are relative imports, not altered architecture dimensions. The curated adapter uses the tested strict checkpoint-loading path and omits unused TalkNet/Dolphin branches.

Dependencies and pretrained models retain their respective upstream licenses; installing a Python package does not change its license or grant rights to its model weights. MediaPipe task sources and official URLs, checkpoint sizes and tested SHA-256 values are in `models/manifest.json`. YOLO26n is obtained through Ultralytics. Furhat SDK and Ollama/Llama are external prerequisites.

No original-project license grant is selected in this initial public repository. Review the complete dependency/model license terms when selecting a project license or redistributing model weights. No third-party model weights or SDK distribution are bundled. The README demonstration screenshot was supplied by the project author for publication.
