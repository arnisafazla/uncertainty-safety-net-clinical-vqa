import pandas as pd
import argparse

from code.utils.evaluation_utils import add_initial_options, final_prediction
from code.utils.analysis_utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--original-folder", required=True, help="Path to the outputs with the original samples. For example ./outputs/GMAI-MMBench/N_210-K_4-M_8")
    parser.add_argument("--plus-nota-folder", required=True, help="Path to the outputs with the original samples. For example ./outputs/GMAI-MMBench/N_210-K_4-M_8-plus_none")
    parser.add_argument("--minus-correct-plus-nota-folder", type=str, required=True, help="Path to the outputs with the original samples. For example ./outputs/GMAI-MMBench/N_210-K_4-M_8-correct_to_none")

    args = parser.parse_args()
    print(vars(args))
    return args

def majority_voting_df(df_default):
    extracted_answers_majority_voting = []
    correct = []
    sample_idx = []
    shuffle_idx = []
    
    for idx, sample_group in df_default.groupby(["sample_idx", "shuffle_idx"], sort=False):
        sample_idx.append(sample_group.iloc[0]['sample_idx'])
        shuffle_idx.append(sample_group.iloc[0]['shuffle_idx'])
        final_pred, is_correct = final_prediction(sample_group)
        extracted_answers_majority_voting.append(final_pred)
        correct.append(is_correct)

    return pd.DataFrame({
        "sample_idx": sample_idx,
        "shuffle_idx": shuffle_idx,
        "extracted_answers_majority_voting": extracted_answers_majority_voting,
        "correct": correct
    })



def main(args: argparse.Namespace):
    MODELS = ["medgemma-4b-it","qwen_2_vl_7b_instruct", "Qwen2.5-VL-7B-Instruct", "Molmo-7B-D-0924", 
        "llava_1.6_7b", "LLaVA-OneVision-1.5-8B-Instruct", 
        "medgemma-27b-it", "Qwen2.5-VL-32B-Instruct", "llava_1.6_34b", 
        "Molmo-72B-0924", "Qwen2.5-VL-72B-Instruct", "gpt-4.1-mini"]

    for MODEL_NAME in MODELS:
        df_default = load_data_direct(MODEL_NAME, f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/extracted.tsv", load_embeddings=[])
        df_plus = load_data_direct(MODEL_NAME, f"{args.plus_nota_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/extracted.tsv", replace_token="plus_none", load_embeddings=[])
        df_minus = load_data_direct(MODEL_NAME, f"{args.minus_correct_plus_nota_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/extracted.tsv", replace_token="correct_to_none", load_embeddings=[])

        df_default_mv = majority_voting_df(df_default)
        df_plus_mv = majority_voting_df(df_plus)
        df_minus_mv = majority_voting_df(df_minus)

        originally_correct = df_default_mv[df_default_mv["correct"]]
        originally_wrong = df_default_mv[~df_default_mv["correct"]]
        print(f"originally_correct: {len(originally_correct)}, originally_wrong: {len(originally_wrong)}")
        originally_correct[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/originally_correct.csv", index=False)
        originally_wrong[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/originally_wrong.csv", index=False)

        df_combined = pd.merge(df_default_mv, df_plus_mv, on=["sample_idx", "shuffle_idx"], suffixes=('_default', '_plus'))
        plus_none_c_to_c = df_combined[df_combined["correct_default"] & df_combined["correct_plus"]]
        plus_none_c_to_w = df_combined[df_combined["correct_default"] & ~df_combined["correct_plus"]]
        plus_none_w_to_w = df_combined[~df_combined["correct_default"] & ~df_combined["correct_plus"]]
        plus_none_w_to_c = df_combined[~df_combined["correct_default"] & df_combined["correct_plus"]]

        plus_none_c_to_w[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/plus_none_c_to_w.csv", index=False)
        plus_none_c_to_c[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/plus_none_c_to_c.csv", index=False)
        plus_none_w_to_w[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/plus_none_w_to_w.csv", index=False)
        plus_none_w_to_c[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/plus_none_w_to_c.csv", index=False)

        df_combined = pd.merge(df_default_mv, df_minus_mv, on=["sample_idx", "shuffle_idx"], suffixes=('_default', '_minus'))
        correct_to_none_c_to_c = df_combined[df_combined["correct_default"] & df_combined["correct_minus"]]
        correct_to_none_c_to_w = df_combined[df_combined["correct_default"] & ~df_combined["correct_minus"]]
        correct_to_none_w_to_w = df_combined[~df_combined["correct_default"] & ~df_combined["correct_minus"]]
        correct_to_none_w_to_c = df_combined[~df_combined["correct_default"] & df_combined["correct_minus"]]

        correct_to_none_c_to_w[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/correct_to_none_c_to_w.csv", index=False)
        correct_to_none_c_to_c[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/correct_to_none_c_to_c.csv", index=False)
        correct_to_none_w_to_w[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/correct_to_none_w_to_w.csv", index=False)
        correct_to_none_w_to_c[["sample_idx", "shuffle_idx"]].to_csv(f"{args.original_folder}/{MODEL_NAME}-default-temp_0.6-topp_0.9-S_10/correct_to_none_w_to_c.csv", index=False)