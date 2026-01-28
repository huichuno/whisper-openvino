import openvino_genai
import librosa
import time
from dotenv import load_dotenv
import os

load_dotenv()

model_id = os.getenv('MODEL_ID')
model_dir = os.getenv('MODEL_DIR')
out_dir = os.path.join(model_dir, model_id)

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

pipe_cpu = openvino_genai.WhisperPipeline(out_dir, "CPU")
pipe_gpu = openvino_genai.WhisperPipeline(out_dir, "GPU")
pipe_auto = openvino_genai.WhisperPipeline(out_dir, "AUTO")

raw_speech = read_wav('audio/audio.mp3')

print("Transcribing using model from %s\n" % out_dir)

# CPU
start = time.time()
result = pipe_cpu.generate(raw_speech)
end = time.time()
elapsed_time = end - start
print(result)
print(f"Elapsed time (CPU): {elapsed_time:.3f} seconds\n") 

# GPU
start = time.time()
result = pipe_gpu.generate(raw_speech)
end = time.time()
elapsed_time = end - start
print(result)
print(f"Elapsed time (GPU): {elapsed_time:.3f} seconds\n")

# AUTO
start = time.time()
result = pipe_auto.generate(raw_speech)
end = time.time()
elapsed_time = end - start
print(result)
print(f"Elapsed time (AUTO): {elapsed_time:.3f} seconds\n") 
