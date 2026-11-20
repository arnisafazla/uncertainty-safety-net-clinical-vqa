import os, io, base64
import json
from typing import Dict, List, Any
from PIL import Image
import pandas as pd
from tqdm import tqdm

from code.utils.gmai_data import format_and_upsample_data, batch_generator

BATCH_SIZE = 300   # IMPORTANT: shard size (500–2000 recommended)
N = 210
K = 4
M = 8
S = 10
# REPLACE_OPTION = "none_of_the_options_are_correct_added"
REPLACE_OPTION = None
# MODEL_NAME = "gemini-2.5-flash-lite"
MODEL_NAME = "gpt-4.1"
# MODEL_NAME = "gpt-5.2"
temperature = 0.6
top_p = 0.9
max_new_tokens = 16

BASE_FOLDER = "/GMAI-MMBench"
INPUT_FILE = "filtered_data.tsv"
IMAGES_DIR = "/projects/prjs1628/MedVQA/GMAI-MMBench/images"
MODEL_MAPPING_PATH = f"{BASE_FOLDER}/model_mapping.json"

if REPLACE_OPTION:
    INPUT_FOLDER = f"{BASE_FOLDER}/N_{N}-K_{K}-M_{M}-{REPLACE_OPTION}"
else:
    INPUT_FOLDER = f"{BASE_FOLDER}/N_{N}-K_{K}-M_{M}"

input_tsv = f"{INPUT_FOLDER}/{INPUT_FILE}"

FINAL_OUTPUT_DIR = (
    f"{INPUT_FOLDER}/{MODEL_NAME}-default-temp_{temperature}-topp_{top_p}-S_{S}"
)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

jsons_dir = os.path.join(FINAL_OUTPUT_DIR, "jsons")
os.makedirs(jsons_dir, exist_ok=True)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_gpt_batch_jsonl(
        batch: List[Dict[str, Any]],
        output_path: str
    ) -> None:
    with open(output_path, "w") as f:
        for idx, item in enumerate(batch):
            sample_idx = item.get("sample_idx", idx)
            shuffle_idx = item["shuffle_idx"]
            generation_idx = item["generation_idx"]

            record = {
                "custom_id": f"{sample_idx}_{shuffle_idx}_{generation_idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL_NAME,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_new_tokens,
                    "include": ["message.output_text.logprobs"],
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": item["prompt"]
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/jpeg;base64,{encode_image(item['image_path'])}",
                                },
                            ]
                        }
                    ]
                }
            }
            if MODEL_NAME in ["gpt-5.2", "gpt-5-mini"]:
                # record["body"]["text"] = {
                #                         "verbosity": "low"
                #                         }
                record["body"]["reasoning"] = {
                                            "effort": "none"
                                        }
            f.write(json.dumps(record) + "\n")

def create_gemini_batch_jsonl(batch: List[Dict[str, Any]], output_path: str):
    """
    Create a JSONL file for Gemini batch using `generate_content` style.
    
    Each batch item should have:
        - prompt: str
        - image_path: str (JPEG)
        - optional sample_idx, shuffle_idx, generation_idx
    """
    with open(output_path, "w") as f:
        for idx, item in enumerate(batch):
            sample_idx = item.get("sample_idx", idx)
            shuffle_idx = item.get("shuffle_idx", 0)
            generation_idx = item.get("generation_idx", 0)
            custom_id = f"{sample_idx}_{shuffle_idx}_{generation_idx}"

            record = {
                "key": custom_id,
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "inlineData": {
                                        "mimeType": "image/jpeg",
                                        "data": encode_image(item["image_path"])
                                    }
                                },
                                {
                                    "text": item["prompt"]
                                }
                            ]
                        }
                    ],
                    "generation_config":{
                        "temperature": temperature,
                        "topP": top_p,
                        "maxOutputTokens": max_new_tokens,
                        "responseLogprobs": True,
                        "logprobs": 1
                    }
                }
            }

            f.write(json.dumps(record) + "\n")

try:
    df = pd.read_csv(input_tsv, sep="\t")
except FileNotFoundError:
    raise RuntimeError(f"File not found: {input_tsv}")

df["image_path"] = df["image_path"].apply(
    lambda x: os.path.join(IMAGES_DIR, x) if pd.notna(x) else ""
)

data: List[Dict[str, Any]] = df.to_dict(orient="records")
processed_data: List[Dict[str, Any]] = format_and_upsample_data(
    data,
    sample_s=S,
    prompt_template="default"
)
print(f"Total processed samples: {len(processed_data)}")

for shard_idx, batch in enumerate(
    tqdm(batch_generator(processed_data, BATCH_SIZE), desc="Creating JSONL shards")
):
    jsonl_path = os.path.join(jsons_dir, f"batch_{shard_idx}.jsonl")

    if "gpt" in MODEL_NAME.lower():
        create_gpt_batch_jsonl(
            batch=batch,
            output_path=jsonl_path
        )
    elif "gemini" in MODEL_NAME.lower():
        create_gemini_batch_jsonl(
            batch=batch,
            output_path=jsonl_path
        )

print(f"Done. JSONL files saved to: {jsons_dir}")
