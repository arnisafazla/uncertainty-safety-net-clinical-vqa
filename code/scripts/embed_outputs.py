import os
import pandas as pd
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", required=True, help="Path to input folder containing inference.tsv")
    args = parser.parse_args()
    print(vars(args))
    return args

def main(args: argparse.Namespace):
    outputs_folder = os.path.join(args.input_folder, "embeddings")
    os.makedirs(outputs_folder, exist_ok=True)
    # embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedding_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

    # 
    for model in ["llava_med_7b", "medgemma-4b-it", "medgemma-27b-it",
            "llava_1.6_7b", "llava_1.6_34b", "LLaVA-OneVision-1.5-8B-Instruct", 
            "qwen_2_vl_7b_instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct", 
            "Molmo-7B-D-0924", "Molmo-72B-0924", "gpt-4.1-mini"]:
        model_folder =  os.path.join(args.input_folder, f"{model}-default-temp_0.6-topp_0.9-S_10")
        if os.path.isdir(model_folder):
            # in each, find the file with the format extracted_DATE_2025.tsv
            extracted_file = [f for f in os.listdir(model_folder) if f.startswith("extracted") and f.endswith(".tsv")]
            if len(extracted_file) != 1:
                print(f"Expected one extracted file in {model_folder}, found {extracted_file}")
                continue
            embedding_file = [f for f in os.listdir(model_folder) if f.endswith("-embeddings-biomedical.npy")]
            if len(embedding_file) > 0:
                print(f"Embeddings already exist for {model}, skipping.")
                continue
            input_tsv_path = os.path.join(model_folder, extracted_file[0])
    
            # Load the input TSV file
            with open(input_tsv_path, "r", encoding="utf-8") as in_file:
                df_inference = pd.read_csv(in_file, sep="\t")

            sentences = df_inference["model_response"].tolist()
            sentences = [str(s) for s in sentences]
            embeddings = embedding_model.encode(sentences, batch_size=32, show_progress_bar=True)
            # output_path = os.path.join(outputs_folder, f"{model}-embeddings.npy")
            output_path = os.path.join(outputs_folder, f"{model}-embeddings-biomedical.npy")
            np.save(output_path, embeddings)
            print(f"Saved embeddings to {output_path}")
        else:
            print(f"Model folder {model_folder} does not exist, skipping.")
            print(os.listdir(args.input_folder))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Extraction completed successfully.")