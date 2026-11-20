import json, os
import pandas as pd
import argparse
import random
from tqdm import tqdm

from code.utils.gmai_data import format_and_upsample_data, batch_generator
from code.utils.transformers_models import InferenceManager

from typing import Dict, List, Any

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input-tsv", required=True, help="Path to GMAI_mm_bench_VAL_images_removed.tsv")
    parser.add_argument("--images-dir", required=True, help="Directory containing images")
    parser.add_argument("--model-mapping-path", type=str, required=True, help="List of modalities to include")
    parser.add_argument("--output-dir", required=True, help="Directory to save the processed TSV")

    parser.add_argument("--model_name", required=True, help="Path to the model directory")
    parser.add_argument("--model_path", default=None, help="Path to the model file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process")
    parser.add_argument("--prompt_template", type=str, default="default", help="Prompt template to use")

    parser.add_argument("--sample_s", type=int, default=10, help="Number of samples to generate per question")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")

    args = parser.parse_args()
    print(vars(args))
    return args

def main(args: argparse.Namespace):
    # DATA PREPARATION
    print(os.getcwd())
    print(os.listdir())

    try:
        with open(args.input_tsv, "r", encoding="utf-8") as in_file:
            df = pd.read_csv(in_file, sep="\t")
    except FileNotFoundError:
        print(f"File {args.input_tsv} not found. Please check the path.")
        print(os.listdir(os.path.dirname(args.input_tsv)))
        return
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(args.images_dir, x) if pd.notna(x) else "")
    data: List[dict] = df.to_dict(orient="records")
    with open(args.model_mapping_path, "r") as in_file:
        models_mapping = json.load(in_file)

    # MODELS SETUP
    model_config = models_mapping[args.model_name]
    model_config["temperature"] = args.temperature
    model_config["top_p"] = args.top_p
    model_config["save_full_distributions"] = True if args.sample_s == 0 else False
    model_id = model_config.get("model_id", None)
    model_type = args.model_name
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(args.output_dir, args.model_name)

    print(f"MODEL ID: {model_id}")
    print(f"MODEL NAME: {args.model_name}")
    print(f"MODEL TYPE: {model_type}")
    
    # Initialize inference manager
    inference_manager = InferenceManager(model_config)
    inference_manager.load_model(model_path)
    
    log_filename = f"inference.tsv"
    
    # Generate data
    processed_data: List[Dict[str, Any]] = format_and_upsample_data(data, sample_s=args.sample_s, prompt_template=args.prompt_template)
    
    # Process batches using the inference manager
    output_rows = []    

    # distributions_folder = os.path.join(args.output_dir, "full_distributions")
    # os.makedirs(distributions_folder, exist_ok=True)

    idx = 0
    for batch in tqdm(batch_generator(processed_data, batch_size=args.batch_size)):
        # print(f"Processing batch of size {len(batch)}", flush=True)
        results: List[Dict[str,Any]] = inference_manager.run_inference(batch)
        
        # Write results
        for output_item in results:
            item = output_item["item"]
            response = output_item["model_response"]
            logprobs = output_item.get("logprobs", [])
            input_logprobs = output_item.get("input_logprobs", [])

            # full_distributions = output_item.get("full_distributions", None)
            # if full_distributions is not None:
            #     dist_filename = f"{idx}.pt"
            #     torch.save(full_distributions, os.path.join(distributions_folder, dist_filename))
            # else:
            #     dist_filename = None

            response_data = {
                "sample_idx": item["sample_idx"],
                "shuffle_idx": item["shuffle_idx"],
                "generation_idx": item["generation_idx"],
                "question": item["question"],
                "prompt": output_item["prompt"],
                "image_path": item["image_path"],
                "original_options":item["original_options"],
                "shuffled_options": item["shuffled_options"],
                "original_correct_answer": item["original_correct_answer"],
                "shuffled_correct_answer": item["shuffled_correct_answer"],
                "difficulty_label": item.get("difficulty_label", "default"),
                "model_response": response,
                "logprobs": logprobs,

                "category": item["category"],
                "clinical VQA task": item["clinical VQA task"],
                "department": item["department"],
                "perceptual granularity": item["perceptual granularity"],
                "modality": item["modality"],
                # "original_task": item["original_task"],
            }

            output_rows.append(response_data)
            idx += 1

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(os.path.join(args.output_dir, log_filename), index=False, sep="\t", encoding="utf-8")

    print(f"Inference complete. Results saved to {log_filename}")

if __name__ == "__main__":
    args = parse_args()    

    random.seed(args.seed)

    main(args)