import json
import math
import pandas as pd
import ast, re
import os
import numpy as np
from typing import List, Dict, Union

from .evaluation_utils import final_prediction
from .metrics import random_accuracy, ECE, ROCAUC, Brier
from .statistical_tests import paired_bootstrap_metric, unpaired_bootstrap_metric

base_path = "code"
with open(f"{base_path}/Uncertainty_MedVQA_scripts/special_tokens.json", "r") as f:
    special_tokens = json.load(f)

def filter_logprobs(logprobs_str_raw, model_name=None):
    logprobs_str = re.sub(
        r'(?<!["\'])-inf(?!["\'])',
        "float('-inf')",
        logprobs_str_raw
    )
    logprobs_list = eval(logprobs_str, {"__builtins__": None}, {"float": float})  # safe eval
    # Replace -inf with a very small number
    logprobs = [(tok, -1e9 if score == float("-inf") else score) for tok, score in logprobs_list]
    if model_name:
        logprobs = [item for item in logprobs if item[0] not in special_tokens.get(model_name, []) + [""]]
    return logprobs

def reverse_map_row(row, replace_token):
    original_options = row["original_options"]
    shuffled_options = row["shuffled_options"]
    model_response = row["extracted_model_response"]
    if model_response == "[]":
        return "[]"
    replaced = "none of the options are correct"
    answer_text = shuffled_options.get(model_response)

    if not answer_text:
        return "[]"

    if not replace_token:
        for key, value in original_options.items():
            if value == answer_text:
                return key
        # assert False, f"Mapping original ({replace_token}): Model response {model_response} with text '{answer_text}' not found in original options {original_options}"

    elif replace_token == "plus_none":
        if answer_text == replaced:
            return "None"
        else:
            for key, value in original_options.items():
                if value == answer_text:
                    return key
        assert False, f"Mapping plus_none ({replace_token}): Model response {model_response} with text '{answer_text}' not found in original options {original_options}"

    elif replace_token == "correct_to_none":
        if answer_text == replaced:
            missing_values = set(original_options.values()) - set(
                v for v in shuffled_options.values() if v != replaced
            )
            if not missing_values:
                assert False, f"Mapping correct_to_none: No missing values found when looking for '{replaced}' in shuffled options {shuffled_options.values()} and original options {original_options.values()}"
            missing_value = missing_values.pop()
            for key, value in original_options.items():
                if value == missing_value:
                    return key
        else: 
            for key, value in original_options.items():
                if value == answer_text:
                    return key
        assert False, f"Mapping correct_to_none ({replace_token}): Model response {model_response} with text '{answer_text}' not found in original options {original_options}"

def load_data(N, K, M, S, MODEL_NAME, PROMPT_TEMPLATE, 
    temperature, top_p, replace_token=None, DATE=None,
    filter_by: dict = {}, filter_samples: List[int] = [], load_embeddings: list = ["embeddings-biomedical"], 
    test=False, base_path="outputs/GMAI-MMBench") -> pd.DataFrame:
    """
    Load extracted and raw data
    """
    
    # Handle DATE being None
    if test:
        base_path = f"{base_path}_test"
   
    if replace_token is not None:
        base_path = f"{base_path}/N_{N}-K_{K}-M_{M}-{replace_token}/"
    else:
        base_path = f"{base_path}/N_{N}-K_{K}-M_{M}/"

    if temperature is not None:
        if S is not None:
            model_path = os.path.join(base_path, f"{MODEL_NAME}-{PROMPT_TEMPLATE}-temp_{temperature}-topp_{top_p}-S_{S}/")
        else:
            model_path = os.path.join(base_path, f"{MODEL_NAME}-{PROMPT_TEMPLATE}-temp_{temperature}-topp_{top_p}/")
    else:
        model_path = os.path.join(base_path, f"{MODEL_NAME}-{PROMPT_TEMPLATE}/")

    if not os.path.exists(model_path):
        model_path = os.path.join(base_path, f"{MODEL_NAME}-{PROMPT_TEMPLATE}-S_{S}/")

    if DATE is None:
        files = os.listdir(model_path)
        inference_files = sorted([f for f in files if f.startswith("inference_") and f.endswith(".tsv")])
        if not inference_files:
            raise FileNotFoundError(f"No extracted file found in {model_path}")
        inference_file = inference_files[-1]
        DATE = inference_file.replace("inference_", "").replace(".tsv", "")
    
    # Load data
    df_extracted = pd.read_csv(os.path.join(model_path, f"extracted.tsv"), sep="\t")
    print(os.path.join(model_path, f"extracted.tsv"), f": {len(df_extracted)}")
    df_extracted["original_options"] = df_extracted["original_options"].apply(ast.literal_eval)
    df_extracted["shuffled_options"] = df_extracted["shuffled_options"].apply(ast.literal_eval)
    df_extracted["extracted_model_response_mapped"] = df_extracted.apply(lambda row: reverse_map_row(row, replace_token), axis=1)
    df_extracted["logprobs"] = df_extracted.apply(lambda row: filter_logprobs(row["logprobs"], model_name=MODEL_NAME), axis=1)

    del df_extracted["shuffled_options"]
    del df_extracted["shuffled_correct_answer"]
    del df_extracted["extracted_model_response"]

    if load_embeddings:
        for embedding_file in load_embeddings:
            # find the single file in model_path with embeddings in name
            embedding_files = [f for f in os.listdir(model_path) if f"{embedding_file}.npy" in f]
            if not embedding_files:
                raise FileNotFoundError(f"No embeddings file found in {model_path}")
            if len(embedding_files) > 1:
                raise ValueError(f"Multiple embeddings files found in {model_path}: {embedding_files}")
            embeddings_path = os.path.join(model_path, embedding_files[0])
            embeddings = np.load(embeddings_path)
            if len(embeddings) != len(df_extracted):
                raise ValueError(f"Embeddings length {len(embeddings)} does not match data length {len(df_extracted)}")
            df_extracted[embedding_file] = list(embeddings)
            # print(f"Loaded embeddings from {embeddings_path}")
    
    for key, value in filter_by.items():
        df_extracted = df_extracted[df_extracted[key] == value]
        print(f"After filtering by {key}={value}: {len(df_extracted)} samples")

    if filter_samples:
        df_extracted = df_extracted[df_extracted["sample_idx"].isin(filter_samples)]
        print(f"After filtering by sample indices: {len(df_extracted)} samples")

    return df_extracted

def load_data_direct(model_name: str, extracted_path: str, replace_token: str = None, load_embeddings: list = ["embeddings-biomedical"], filter=None):
    # Load data
    df_extracted = pd.read_csv(extracted_path, sep="\t")
    df_extracted["original_options"] = df_extracted["original_options"].apply(ast.literal_eval)
    df_extracted["shuffled_options"] = df_extracted["shuffled_options"].apply(ast.literal_eval)
    df_extracted["extracted_model_response_mapped"] = df_extracted.apply(lambda row: reverse_map_row(row, replace_token), axis=1)
    df_extracted["logprobs"] = df_extracted.apply(lambda row: filter_logprobs(row["logprobs"], model_name=model_name), axis=1)

    df_extracted["iloc_index"] = df_extracted.index
    if filter is not None:
        sample_shuffle_indices = filter.to_records(index=False).tolist()
        mask = pd.concat(
            [(df_extracted["sample_idx"] == r) & (df_extracted["shuffle_idx"] == c) for r, c in sample_shuffle_indices],
            axis=1
        ).any(axis=1)
        df_extracted = df_extracted[mask]

    if load_embeddings:
        for embedding_file in load_embeddings:
            # find the single file in model_path with embeddings in name
            model_path = os.path.dirname(extracted_path)
            embedding_files = [f for f in os.listdir(model_path) if f"{embedding_file}.npy" in f]
            if not embedding_files:
                raise FileNotFoundError(f"No embeddings file found in {model_path}")
            if len(embedding_files) > 1:
                raise ValueError(f"Multiple embeddings files found in {model_path}: {embedding_files}")
            embeddings_path = os.path.join(model_path, embedding_files[0])
            embeddings = np.load(embeddings_path)
            embeddings = embeddings[df_extracted["iloc_index"].values]
            if len(embeddings) != len(df_extracted):
                raise ValueError(f"Embeddings length {len(embeddings)} does not match data length {len(df_extracted)}")
            df_extracted[embedding_file] = list(embeddings)
            # print(f"Loaded embeddings from {embeddings_path}")
    
    if replace_token == "plus_none":
        df_extracted["extracted_model_response_mapped"] = df_extracted["extracted_model_response_mapped"].fillna("None")
    return df_extracted

def compute_extraction_details(df_extracted: pd.DataFrame):
    empty_answers = df_extracted[df_extracted['extraction_method'] == 'empty_answer'].shape[0]
    re_search_answers = df_extracted[df_extracted['extraction_method'] == 're_search'].shape[0]
    llm_extracted_total = len(df_extracted[df_extracted['extraction_method'] == "llm_extracted"])
    llm_extracted_failed = int((df_extracted[df_extracted['extraction_method'] == "llm_extracted"]['extracted_model_response_mapped'] == "[]").sum())
    llm_extracted_success = llm_extracted_total - llm_extracted_failed

    return {
        "total_samples": len(df_extracted),
        "empty_answers": empty_answers,
        "empty_answers_ratio": round(empty_answers / len(df_extracted) * 100, 3) if len(df_extracted) > 0 else 0,
        "re_search_answers": re_search_answers,
        "re_search_answers_ratio": round(re_search_answers / len(df_extracted) * 100, 3) if len(df_extracted) > 0 else 0,
        "llm_extracted_success": llm_extracted_success,
        "llm_extracted_success_ratio": round(llm_extracted_success / len(df_extracted) * 100, 3) if len(df_extracted) > 0 else 0,
        "llm_extracted_failed": llm_extracted_failed,
        "llm_extracted_failed_ratio": round(llm_extracted_failed / len(df_extracted) * 100, 3) if len(df_extracted) > 0 else 0,
    }
    
def calculate_sample_level_stats(df_extracted: pd.DataFrame, S = None, load_embeddings: list = ["embeddings-biomedical"]) -> Dict[str, Union[np.ndarray, list]]:
    # return list of correct 1 or incorrect 0
    def get_correctness(pred, gold) -> np.ndarray:
        if len(pred) != len(gold):
            raise ValueError("Predictions and gold labels must have the same length.")
        return np.array([1 if p == g else 0 for p, g in zip(pred, gold)])
    
    # Aggregate
    extracted_answers = df_extracted['extracted_model_response_mapped'].tolist()
    correct_answers = df_extracted['original_correct_answer'].tolist()
    correctness_per_sample = get_correctness(extracted_answers, correct_answers)

    # Get log probabilities
    logprobs = df_extracted['logprobs'].tolist()
    logprobs_list_nums = [[item[1] for item in sample] for sample in logprobs]

    # Get statistics grouped by sample/shuffle
    extracted_answers_per_sample = []
    model_responses_per_sample = []
    extracted_answers_majority_voting = []
    correctness_majority_voting = []
    option_counts = []
    sample_idx = []
    shuffle_idx = []
    logprobs_per_sample = []
    logprobs_list_per_sample = []
    embeddings_per_sample = {
            f"{embedding_file}_per_sample": [] for embedding_file in load_embeddings
    }
    
    # Assign a group id based on integer division of the row number by S
    df_extracted = df_extracted.reset_index(drop=True)

    # Here, flatten all on sample level.
    if S is None:
        group_id = "sample_idx"
    # Here, flatten all on sample+shuffle level.
    else:
        group_id = ["sample_idx", "shuffle_idx"]

    for idx, sample_group in df_extracted.groupby(group_id, sort=False):
        sample_idx.append(sample_group.iloc[0]['sample_idx'])
        shuffle_idx.append(sample_group.iloc[0]['shuffle_idx'])

        extracted_answers_per_sample.append(sample_group['extracted_model_response_mapped'].dropna().tolist())
        model_responses_per_sample.append(sample_group['model_response'].dropna().tolist())
        logprobs_per_sample.append(list(zip(extracted_answers_per_sample[-1], sample_group['logprobs'].tolist())))
        logprobs_list_per_sample.append([[item[1] for item in sample] for sample in sample_group['logprobs'].tolist()])
        for embedding_file in load_embeddings:
            if embedding_file in sample_group.columns:
                embeddings_per_sample[f"{embedding_file}_per_sample"].append(sample_group[embedding_file].dropna().tolist())

        final_pred, is_correct = final_prediction(sample_group)
        extracted_answers_majority_voting.append(final_pred)

        correctness_majority_voting.append(int(is_correct) if is_correct is not None else np.nan)
        option_counts.append(len(sample_group.iloc[0]['original_options'].keys()))        

    return {
        "sample_idx": sample_idx,
        "shuffle_idx": shuffle_idx,
        "extracted_answers_per_sample": extracted_answers_per_sample,
        "model_responses_per_sample": model_responses_per_sample,
        "extracted_answers_majority_voting": extracted_answers_majority_voting,
        "correctness_per_sample": correctness_per_sample,
        "correctness_majority_voting": np.array(correctness_majority_voting),
        "raw_greedy_log_likelihoods": logprobs,
        "greedy_log_likelihoods": logprobs_list_nums,
        "raw_logprobs_per_sample": logprobs_per_sample,
        "logprobs_list_per_sample": logprobs_list_per_sample,
        "options_counts": option_counts,
        **embeddings_per_sample
    }

def compare_across_methods(metric_name: str, all_ue_estimates: Dict[str, Dict[str, List[float]]]):
    comparisons = []
    if metric_name not in all_ue_estimates[next(iter(all_ue_estimates))]:
        print(f"Skipping metric: {metric_name}")
        return comparisons
    metric_values: Dict[str, float] = {estimator: values[metric_name] for estimator, values in all_ue_estimates.items()}
    
    metric = ECE(normalize_min_max=False) if metric_name == "ece" else ROCAUC()
    # rank the methods by auroc
    ranked_methods = sorted(metric_values.items(), key=lambda x: x[1], reverse=False if metric_name == "ece" else True)
    # compute significance between best model and next, until the p-value < 0.05
    if not ranked_methods:
        return comparisons
    
    baseline_method = ranked_methods[0][0]
    # print(baseline_method)
    for method, metric_value in ranked_methods[1:]:
        # print(method)
        targets = np.array(all_ue_estimates[baseline_method]["targets"])
        baseline_estimates = np.array(all_ue_estimates[baseline_method]["estimates"])
        method_estimates = np.array(all_ue_estimates[method]["estimates"])
        # Mask out NaN values
        
        # print(f"    Filtering out NaNs: {len(targets)} samples remain for comparison.")
        bootstrap_results = paired_bootstrap_metric(
            metric, [baseline_estimates, method_estimates], targets
        )
        p_value = bootstrap_results[(0, 1)]['p_value']
        # print(f"  Comparing {baseline_method} (AUROC={metric_values[baseline_method]:.4f}) vs {method} (AUROC={metric_values[method]:.4f}): p-value = {p_value:.4f}")
        # print(bootstrap_results[(0, 1)])
        comparisons.append({
            'method_1': baseline_method,
            'method_2': method,
            f'{metric_name}_1': metric_values[baseline_method],
            f'{metric_name}_2': metric_value,
            'p_value': p_value,
            'bootstrap_results': bootstrap_results[(0, 1)],
        })
        if p_value < 0.05:
            # print("  Significant difference found, stopping comparisons.")
            return comparisons
    return comparisons

def compare_pairwise(metric_name: str, all_ue_estimates: Dict[str, Dict[str, List[float]]]):
    metrics = {
        "auroc": ROCAUC(),
        "ece": ECE(normalize_min_max=False),
        "brier": Brier(normalize_min_max=False),
    }
    metric = metrics[metric_name]
    
    baseline_estimates = np.array(all_ue_estimates["default"]["estimates"])
    modified_estimates = np.array(all_ue_estimates["modified"]["estimates"])
    baseline_targets = np.array(all_ue_estimates["default"]["targets"])
    modified_targets = np.array(all_ue_estimates["modified"]["targets"])
    
    # print(f"    Filtering out NaNs: {len(targets)} samples remain for comparison.")
    bootstrap_results = unpaired_bootstrap_metric(
        metric, modified_estimates, modified_targets, baseline_estimates, baseline_targets
    )   # first is a, second is b
    # print(f"  Comparing {baseline_method} (AUROC={metric_values[baseline_method]:.4f}) vs {method} (AUROC={metric_values[method]:.4f}): p-value = {p_value:.4f}")
    return {
        'default': bootstrap_results["observed_b"],
        'modified': bootstrap_results["observed_a"],
        'p_value': bootstrap_results["p_value"],
        'difference': bootstrap_results["observed_diff"],
    }
    
def aggregate_results(stats, metrics: List[dict]) -> dict:
    results = {}

    correctness_per_sample = stats["correctness_per_sample"]
    correctness_majority_voting = stats["correctness_majority_voting"]

    for group in metrics:
        estimator = group["ue_method"]
        estimator_name = str(estimator)
        estimator_outputs = estimator(stats)
        
        if len(estimator_outputs) == len(correctness_per_sample):
            target = correctness_per_sample
        elif len(estimator_outputs) == len(correctness_majority_voting):
            target = correctness_majority_voting
        else:
            raise ValueError(f"Length mismatch between estimator outputs ({len(estimator_outputs)}) and available targets ({len(correctness_per_sample)} or {len(correctness_majority_voting)}).")

        estimator = np.asarray(estimator_outputs, dtype=float)
        target = np.asarray(target, dtype=int)

        # Mask out NaN values
        mask = ~np.isnan(estimator)
        estimator = estimator[mask]
        target = target[mask]

        if len(estimator) == 0:
            raise ValueError(f"{estimator_name}: All estimator values are NaN.")
        # else:
        #     print(f"Computing Brier score on {len(estimator)} valid samples ({estimator_name}).")

        for ue_metric in group["metrics"]:
            results[f"{estimator_name}_{str(ue_metric)}"] = ue_metric(estimator, target)       
        
    results["avg_accuracy_majority_voting"] = round(correctness_majority_voting.mean(), 3)
    results["avg_accuracy"] = round(correctness_per_sample.mean(), 3)

    return results

def min_max_normalize(model_name, estimator_name, method_estimates, min_max: str = "per_model"):
    # Min-max normalization -ignore nan values
    if min_max == "global":
        min_val = 0.0
        max_vals = {'LabelNLL': np.float64(7.021522),
            'ANLL': np.float64(2.33349),
            'MaxNLL': np.float64(8.64729),
            'SC': np.float64(0.8),
            'SE': np.float64(1.609438),
            'PRO': np.float64(3.67852),
            'EE_embeddings-biomedical': np.float64(144.360687),
            'RDS_embeddings-biomedical': np.float64(132.276428)}
        max_val = max_vals.get(estimator_name)
    elif min_max == "per_model":
        min_val = 0.0
        with open(f"{base_path}/plots/percentile_values.json", "r") as f:
            max_vals = json.load(f)
        max_val = max_vals.get(model_name, {}).get(estimator_name, 1.0)
    else:
        min_val = np.nanmin(method_estimates)
        max_val = np.nanmax(method_estimates)
    if max_val - min_val > 0:
        method_estimates = (method_estimates - min_val) / (max_val - min_val)
    else:
        method_estimates = method_estimates - min_val  # all values are the same

    return np.clip(method_estimates, 0.0, 1.0)

def aggregate_results_with_test(stats, metrics: List[dict], model_name: str, min_max: str = "per_model") -> dict:
    def pm_ci(correctness, z=1.96):
        n = len(correctness)
        p_hat = sum(correctness) / n

        denom = 1 + z**2 / n
        center = p_hat + z**2 / (2 * n)
        margin = z * math.sqrt(
            (p_hat * (1 - p_hat) + z**2 / (4 * n)) / n
        )

        lower = (center - margin) / denom
        upper = (center + margin) / denom

        pm = (upper - lower) / 2
        return p_hat, pm

    results = {}
    correctness_per_sample = stats["correctness_per_sample"]
    correctness_majority_voting = stats["correctness_majority_voting"]

    all_ue_estimates = {}
    for group in metrics:
        estimator = group["ue_method"]
        estimator_name = str(estimator)
        estimator_outputs = estimator(stats)
        
        if len(estimator_outputs) == len(correctness_per_sample):
            target = correctness_per_sample
        elif len(estimator_outputs) == len(correctness_majority_voting):
            target = correctness_majority_voting
        else:
            raise ValueError(f"Length mismatch between estimator outputs ({len(estimator_outputs)}) and available targets ({len(correctness_per_sample)} or {len(correctness_majority_voting)}).")

        estimator_outputs = np.asarray(estimator_outputs, dtype=float)
        target = np.asarray(target, dtype=int)
        
        all_ue_estimates[estimator_name] = {}
        all_ue_estimates[estimator_name]["estimates"] = estimator_outputs
        all_ue_estimates[estimator_name]["targets"] = target

    all_method_estimates = np.array([values["estimates"] for estimator, values in all_ue_estimates.items()])
    # mask out samples where any method has NaN
    mask = ~np.isnan(all_method_estimates).any(axis=0)
    # print(f"Mask shape: {mask.shape}, {mask.sum()} samples remain after filtering out NaNs across all methods.")
    for group in metrics:
        estimator = group["ue_method"]
        estimator_name = str(estimator)
        # print(estimator_name)
        method_estimates = np.array(all_ue_estimates[estimator_name]["estimates"])
        targets = np.array(all_ue_estimates[estimator_name]["targets"])
        # Apply mask
        method_estimates = method_estimates[mask]
        targets = targets[mask]
        min_max_norm = group["metrics"][0].normalize_min_max
        if min_max_norm:
            method_estimates = min_max_normalize(model_name, estimator_name, method_estimates, min_max=min_max)

        all_ue_estimates[estimator_name]["estimates"] = method_estimates.tolist()
        all_ue_estimates[estimator_name]["targets"] = targets.tolist()

        if len(method_estimates) == 0:
            raise ValueError(f"{estimator_name}: All estimator values are NaN.")

        for ue_metric in group["metrics"]:
            all_ue_estimates[estimator_name][str(ue_metric)] = ue_metric(method_estimates, targets)
            results[f"{estimator_name}_{str(ue_metric)}"] = all_ue_estimates[estimator_name][str(ue_metric)]

    comparisons_AUROC, comparisons_ECE = [], []
    if metrics and metrics[0]["metrics"]:
        comparisons_AUROC: list = compare_across_methods("auroc", all_ue_estimates)
        comparisons_ECE: list = compare_across_methods("ece", all_ue_estimates)

    avg_acc, pm = pm_ci(correctness_majority_voting)
    results["avg_accuracy_majority_voting"] = round(avg_acc*100, 1)
    results["pm_accuracy_majority_voting"] = round(pm*100, 1)
    
    results["avg_accuracy"] = round(correctness_per_sample.mean(), 3)

    return results, comparisons_AUROC, comparisons_ECE

def summarize_extraction(N, K, M, S, MODEL_NAME, PROMPT_TEMPLATE, temperature, top_p, replace_token=None, DATE=None,
                         metrics=[],
                         sample_K = None,
                         per: Union[str, None] = None,
                         filter_by: dict = {},
                         filter_samples: List[int] = [], load_embeddings: list = ['embeddings-biomedical'],
                         test: bool = False,
                         min_max: str = "per_model") -> dict:
    """
    Load extracted and raw data, compute extraction and accuracy metrics,
    apply custom metrics (e.g., consistency, entropy), and return a single-row summary DataFrame.
    """
    def compute_all_results(df, K, S, metrics, load_embeddings=None):
        # Extraction details
        extraction_details_results = compute_extraction_details(df)

        df = df[df["extracted_model_response_mapped"].notna()]
        df = df[df['extracted_model_response_mapped'] != "[]"]
        print(f"Len df after removing NaN and empty responses = {len(df)}")
        
        options = df['original_options'].tolist()
        options = [len(opt.keys()) for opt in options]
            
        # Compute metrics per group
        S = None if K == 1 else S
        stats = calculate_sample_level_stats(df, S=S, load_embeddings=load_embeddings)
        # try:
        ue_results, comparisons_AUROC, comparisons_ECE = aggregate_results_with_test(stats, metrics=metrics, model_name=MODEL_NAME, min_max=min_max)
        # except Exception as e:
        #     print("Error in aggregate_results:", e)
        #     return None, stats

        # Build summary DataFrame
        results = {
            "N": N,
            "K": K,
            "M": M,
            "S": S,
            f"{per}": df[per].iloc[0] if per else "all",
            "temperature": temperature,
            "top_p": top_p,
            "MODEL_NAME": MODEL_NAME,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
            "DATE": DATE,
            "filtered_random_accuracy": round(random_accuracy(options), 3)
        }

        results.update(extraction_details_results)
        results.update(ue_results)

        return results, comparisons_AUROC, comparisons_ECE 

    ue_methods = [metric.get("ue_method") for metric in metrics if metric.get("ue_method") is not None]
    df = load_data(N, K, M, S, MODEL_NAME, PROMPT_TEMPLATE, temperature, top_p, 
        replace_token=replace_token, DATE=DATE, filter_by=filter_by, filter_samples=filter_samples,
        load_embeddings=load_embeddings, test=test)

    if sample_K is not None:
        df = df.groupby("sample_idx", group_keys=False).sample(sample_K, random_state=42)
        print(f"Len df after sampling K={sample_K} = {len(df)}")

    if per:
        assert per in ["modality", "perceptual granularity", "department", 'clinical VQA task', "category"], "Invalid value for 'per'. Must be 'modality', 'granularity', or 'department'."
        if per == "clinical VQA task":
            def group_tasks(x):
                if "Recognition" in x or x in ["Bone", "Muscle"]:
                    return "Recognition"
                elif "Grading" in x:
                    return "Grading"
                elif "Diagnosis" in x:
                    return "Diagnosis"
                else:
                    return "Other"
            df["clinical VQA task"] = df["clinical VQA task"].apply(group_tasks)
        # split df into multiple dataframes based on modality
        categories = df[per].unique()
        results_list = []
        sig_results_list = []
        num_samples = {category: len(df[df[per] == category]["sample_idx"].unique()) for category in categories}
        for category in categories:
            df_category = df[df[per] == category]
            print(f"Processing {per}: {category} with {len(df_category)} samples")
            if len(df_category) == 0:
                continue
            results, comparisons_AUROC, comparisons_ECE = compute_all_results(df_category, K, S, metrics, load_embeddings=load_embeddings)
            significant_methods = [comparison['method_2'] for comparison in comparisons_AUROC if comparison['p_value'] >= 0.05]
            for method in ue_methods:
                method_name = str(method)
                if method_name == comparisons_AUROC[0]['method_1']:
                    sig_results_list.append(f'{method_name},{category},"AUROC",1')
                elif method_name in significant_methods:
                    sig_results_list.append(f'{method_name},{category},"AUROC",1')
                else:
                    sig_results_list.append(f'{method_name},{category},"AUROC",0')
            significant_methods = [comparison['method_2'] for comparison in comparisons_ECE if comparison['p_value'] >= 0.05]
            for method in ue_methods:
                method_name = str(method)
                if method_name == comparisons_ECE[0]['method_1']:
                    sig_results_list.append(f'{method_name},{category},"ECE",1')
                elif method_name in significant_methods:
                    sig_results_list.append(f'{method_name},{category},"ECE",1')
                else:
                    sig_results_list.append(f'{method_name},{category},"ECE",0') 
                
            if results is not None:
                results["type"] = category
                results_list.append(results)
                # stats_list.append(stats)
    
    else:
        results, comparisons_AUROC, comparisons_ECE = compute_all_results(df, K, S, metrics, load_embeddings=load_embeddings)
        results_list = [results] if results is not None else []
        sig_results_list = []
        num_samples = len(df["sample_idx"].unique())
        if not comparisons_AUROC:
            return results_list, sig_results_list, {"total_samples": num_samples}
        significant_methods = [comparison['method_2'] for comparison in comparisons_AUROC if comparison['p_value'] >= 0.05]
        if len(ue_methods) > 1:
            for method in ue_methods:
                method_name = str(method)
                if method_name == comparisons_AUROC[0]['method_1']:
                    sig_results_list.append(f'{method_name},{MODEL_NAME},"AUROC",1')
                elif method_name in significant_methods:
                    sig_results_list.append(f'{method_name},{MODEL_NAME},"AUROC",1')
                else:
                    sig_results_list.append(f'{method_name},{MODEL_NAME},"AUROC",0')
            significant_methods = [comparison['method_2'] for comparison in comparisons_ECE if comparison['p_value'] >= 0.05]
            for method in ue_methods:
                method_name = str(method)
                if method_name == comparisons_ECE[0]['method_1']:
                    sig_results_list.append(f'{method_name},{MODEL_NAME},"ECE",1')
                elif method_name in significant_methods:
                    sig_results_list.append(f'{method_name},{MODEL_NAME},"ECE",1')
                else:
                    sig_results_list.append(f'{method_name},{MODEL_NAME},"ECE",0') 
    return results_list, sig_results_list, num_samples