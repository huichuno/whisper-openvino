# whisper-openvino
This example showcases how to export and inference of speech recognition Whisper Models using OpenVINO

## Prerequisites
* git - distributed version control system
    * [Windows](https://git-scm.com/install/windows) (validated)
    * [Linux](https://git-scm.com/install/linux)
* uv - fast python package manager
    * [Windows](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2) (validated)
    * [Linux](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)

## Supported Hardware
* Intel® Core™ Ultra Processors (Series 2) (validated)

## Getting Started
```sh
git clone https://github.com/huichuno/whisper-openvino.git

cd whisper-openvino

uv sync
```
## Usage

### Export Whisper model to OpenVINO IR format
```
uv run .\export_whisper.py

# check out 'models\' folder for exported models
# update configuration in '.env' file as needed
```

### Run exported Whisper model
```
uv run .\run_whisper.py

# Output:
# Oh, you think darkness is your ally? You merely adopted the dark. I was born in it. Molded by it. I didn't see the light until I was already a man. By then it was nothing to me but BLINDED!
# Elapsed time (CPU): 12.612 seconds

# Oh, you think darkness is your ally? You merely adopted the dark. I was born in it. Molded by it. I didn't see the light until I was already a man. By then it was nothing to me but BLINDED!
# Elapsed time (GPU): 1.574 seconds

# Oh, you think darkness is your ally? You merely adopted the dark. I was born in it. Molded by it. I didn't see the light until I was already a man. By then it was nothing to me but BLINDED!
# Elapsed time (AUTO): 1.429 seconds
```

### Alternate method to export Whisper model to OpenVINO IR format
```sh
uv run optimum-cli export openvino --model openai/whisper-large-v3 models/openai/whisper-large-v3
```
## Reference
* https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/whisper_speech_recognition
* https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/export-requirements.txt
* https://docs.astral.sh/uv/