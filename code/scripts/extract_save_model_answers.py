import json, os
import pandas as pd
import argparse
from code.utils.evaluation_utils import EvaluateModelOutputs

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input-tsv", required=True, help="Path to inference.tsv")
    parser.add_argument("--output-dir", required=True, help="Directory to save the processed TSV")
    parser.add_argument("--model-mapping-path", type=str, required=True, help="Path to the model mapping JSON file")
    parser.add_argument("--model_name", required=True, help="Path to the model directory")

    args = parser.parse_args()
    print(vars(args))
    return args

def main(args: argparse.Namespace):
    # Load the input TSV file
    with open(args.input_tsv, "r", encoding="utf-8") as in_file:
        df_inference = pd.read_csv(in_file, sep="\t")
    
    # MODELS SETUP
    with open(args.model_mapping_path, "r") as in_file:
        models_mapping = json.load(in_file)
    model_config = models_mapping[args.model_name]
    model_id = model_config.get("model_id", None)
    
    # Initialize the evaluation class
    evaluator = EvaluateModelOutputs(model_id=model_id, model_config=model_config)

    # Extract results and save them
    output_path = os.path.join(args.output_dir, "extracted.tsv")
    evaluator.extract_results_pipeline(df_inference, output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Extraction completed successfully.")