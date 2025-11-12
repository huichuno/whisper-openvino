import openvino_genai
import librosa
import time

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

model_dir = "models/whisper-large-v3"

pipe_cpu = openvino_genai.WhisperPipeline(model_dir, "CPU")
pipe_gpu = openvino_genai.WhisperPipeline(model_dir, "GPU")
pipe_auto = openvino_genai.WhisperPipeline(model_dir, "AUTO")

raw_speech = read_wav('audio/how_r_u.wav')

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
