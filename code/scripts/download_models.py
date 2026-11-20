#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

MODELS = {
    "llava_1.6_7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "deepseek_1.3b": "deepseek-ai/deepseek-vl-1.3b-base",
    "llava_1.6_34b": "llava-hf/llava-v1.6-34b-hf",
    "Molmo-72B-0924": "allenai/Molmo-72B-0924",
    "Molmo-7B-D-0924": "allenai/Molmo-7B-D-0924",
    "medgemma-4b-it": "google/medgemma-4b-it",
    "medgemma-27b-it": "google/medgemma-27b-it",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "LLaVA-OneVision-1.5-8B-Instruct": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
}

def download_models(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, repo_id in MODELS.items():
        model_dir = os.path.join(output_dir, name)
        if os.path.exists(model_dir):
            print(f"Model {name} already exists at {model_dir}, skipping download.")
            continue
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading {repo_id} into {model_dir}")
        snapshot_download(repo_id=repo_id, local_dir=model_dir)
        
def main(out_dir: str):
    # Authenticate with Hugging Face Hub
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env or environment variables.")
    login(token=token)
    download_models(out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download multiple HuggingFace models locally")
    parser.add_argument("-o", "--output-dir", required=True, help="Where to save downloaded models")
    args = parser.parse_args()

    args = parser.parse_args()
    print(vars(args))
    main(args.output_dir)
