from optimum.exporters.openvino.convert import export_tokenizer
from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer, AutoProcessor, AutoFeatureExtractor
import os

out_dir = "models/whisper-large-v3"
model_id = "openai/whisper-large-v3"

try:
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        trust_remote_code=True,
    )
    model.save_pretrained(out_dir)
    print(f"model saved to {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(out_dir)
    export_tokenizer(tokenizer, out_dir)
    print(f"tokenizer saved to {out_dir}")

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        processor.save_pretrained(out_dir)
        print(f"processor saved to {out_dir}")
    except Exception as e:
        print(f"Processor save warning: {e}\n")
    
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        feature_extractor.save_pretrained(out_dir)
        print(f"feature extractor saved to {out_dir}")
    except Exception as e:
        print(f"feature extractor save warning: {e}\n")

except Exception as e:
    print(f"Error during export: {e}\n")
    raise