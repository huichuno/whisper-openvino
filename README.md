# whisper-openvino
This example showcases inference of speech recognition Whisper Models on OpenVINO

## How Export Whisper model to OpenVINO format
```sh
uv run optimum-cli export openvino --model openai/whisper-large-v3 models/whisper-large-v3
```
