import huggingface_hub as hf_hub
from dotenv import load_dotenv
import os

load_dotenv()

model_id = os.getenv('MODEL_ID')
model_dir = os.getenv('MODEL_DIR')
out_dir = os.path.join(model_dir, model_id)

print(f"Downloading model {model_id} to {out_dir}...")
hf_hub.snapshot_download(model_id, local_dir=out_dir)