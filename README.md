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
# update configuration in *.env* file as needed
```

### Run exported Whisper model
```
uv run .\run_whisper.py

# Output:
# How are you doing today?
# Elapsed time (CPU): 8.089 seconds

# How are you doing today?
# Elapsed time (GPU): 1.194 seconds

# How are you doing today?
# Elapsed time (AUTO): 0.744 seconds
```

### Alternate method to export Whisper model to OpenVINO IR format
```sh
uv run optimum-cli export openvino --model openai/whisper-large-v3 models/openai/whisper-large-v3
```
## Reference
* https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/whisper_speech_recognition
* https://github.com/openvinotoolkit/openvino.genai/blob/master/samples/export-requirements.txt
* https://docs.astral.sh/uv/