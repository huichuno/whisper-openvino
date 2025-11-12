from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer, AutoProcessor, AutoFeatureExtractor
from dotenv import load_dotenv
import os

load_dotenv()

# model_id = "openai/whisper-large-v3"
# out_dir = "models/whisper-large-v3"

model_id = os.getenv('MODEL_ID')
model_dir = os.getenv('MODEL_DIR')
out_dir = os.path.join(model_dir, model_id)


try:
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        trust_remote_code=True,
    )
    model.save_pretrained(out_dir)
    print(f"Model saved to {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)
    export_tokenizer(tokenizer, out_dir)
    print(f"Tokenizer saved to {out_dir}")

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(out_dir)
        print(f"Processor saved to {out_dir}")

    except Exception as e:
        print(f"Processor save warning: {e}\n")
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        feature_extractor.save_pretrained(out_dir)
        print(f"Feature extractor saved to {out_dir}")

    except Exception as e:
        print(f"Feature extractor save warning: {e}\n")

except Exception as e:
    print(f"Error during export: {e}\n")
    raise