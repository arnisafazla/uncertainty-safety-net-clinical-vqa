import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the saved values in plots folder, then generate the latex table and save it in plots folder"
    )
    parser.add_argument(
        "--input-folder", "-f", required=True,
        help="Path to the main plots folder, for example outputs/GMAI-MMBench/plots/deltas"
    )
    parser.add_argument(
            "--output-folder", "-f", required=True,
            help="Path to the output plots folder, for example ./plots_and_tables"
        )
    
    args = parser.parse_args()
    print(vars(args))
    return args

def main(args):
    # Load CSV
    N = 210 # Number of samples per modality
    K = 4   # Number of shuffles per sample
    M = 8   # Number of modalities
    S = 10  # Number of repeated samplings per each shuffle
    os.makedirs(args.output_folder, exist_ok=True)

    MODEL_NAME_MAPPING = {
        "llava_med_7b": "LLaVA-Med-7B",
        "medgemma-4b-it": "MedGemma-4B",
        "medgemma-27b-it": "MedGemma-27B",
        "llava_1.6_7b": "LLaVA-v1.6-7B",
        "llava_1.6_34b": "LLaVA-v1.6-34B",
        "LLaVA-OneVision-1.5-8B-Instruct": "LLaVA-OV-8B",
        "qwen_2_vl_7b_instruct": "Qwen2-VL-7B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
        "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
        "Molmo-7B-D-0924": "Molmo-7B",
        "Molmo-72B-0924": "Molmo-72B",
        "gpt-4.1-mini": "GPT-4.1-Mini",
    }
    MODELS = ["medgemma-4b-it", "qwen_2_vl_7b_instruct", "Qwen2.5-VL-7B-Instruct", "Molmo-7B-D-0924",
    "llava_med_7b", "llava_1.6_7b", "LLaVA-OneVision-1.5-8B-Instruct", "medgemma-27b-it", "Qwen2.5-VL-32B-Instruct",
    "llava_1.6_34b", "Molmo-72B-0924", "Qwen2.5-VL-72B-Instruct", "gpt-4.1-mini"]
    ESTIMATORS = ["LabelNLL", "ANLL", "MaxNLL", "SC", "SE", "PRO", "EE", "RDS"]

    # SELECT WHICH METRICS TO INCLUDE IN THE TABLE!!
    METRICS = ["AUROC", "Brier", "Avg"]
    METRICS = ["Accuracy", "ECE"]
    METRIC_DISPLAY = {
        "Accuracy": "Accuracy",
        "AUROC": "AUROC",
        "ECE": "ECE",
        "Brier": "Brier",
        "Avg": "H", 
    }
    LOWER_IS_BETTER = {"ECE", "Brier"}

    df_results = defaultdict(dict)
    for metric in METRICS:
        df_results[metric] = {}
        for model_name in MODELS:
            df_results[metric][model_name] = {"plus_none": {}, "correct_to_none": {}}

    values_all = []
    for estimator_name in ESTIMATORS:
        for model_name in MODELS:
            file_path = f"{args.input_folder}/{estimator_name}/{model_name}_results.json"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    tmp_results = json.load(f)
                for metric in METRICS:
                    df_results[metric][model_name]["plus_none"][estimator_name] = tmp_results["plus_none"][metric]
                    df_results[metric][model_name]["correct_to_none"][estimator_name] = tmp_results["correct_to_none"][metric]
                    values_all.append(tmp_results["plus_none"][metric]["difference"])
                    values_all.append(tmp_results["correct_to_none"][metric]["difference"])
            else:
                for metric in METRICS:
                    df_results[metric][model_name]["plus_none"][estimator_name] = {"difference": np.nan, "p_value": np.nan}
                    df_results[metric][model_name]["correct_to_none"][estimator_name] = {"difference": np.nan, "p_value": np.nan}   
    print(df_results)

    ALPHA = 0.05

    BLUE_RGB   = np.array([0, 114, 178])   / 255.0
    ORANGE_RGB = np.array([230, 159, 0])   / 255.0
    WHITE_RGB  = np.array([255, 255, 255]) / 255.0

    BLUE_ORANGE_CMAP = LinearSegmentedColormap.from_list(
        "blue_orange",
        [ORANGE_RGB, WHITE_RGB, BLUE_RGB]  # negative → zero → positive
    )

    GOLD_RGB   = np.array([191, 129, 45])  / 255.0  # muted gold (not orange)
    PURPLE_RGB = np.array([120, 94, 181])  / 255.0  # soft purple
    WHITE_RGB  = np.array([255, 255, 255]) / 255.0

    GOLD_PURPLE_CMAP = LinearSegmentedColormap.from_list(
        "gold_purple",
        [GOLD_RGB, WHITE_RGB, PURPLE_RGB]  # negative → zero → positive
    )

    # -----------------------
    # Helpers
    # -----------------------
    def format_value(val):
        sign = "+" if val > 0 else ""
        return f"{sign}{val * 100:.1f}"

    lim_for_coloring_metrics = max(abs(np.min(values_all)), abs(np.max(values_all)))
    lim_for_coloring_avg = max(abs(np.min(values_all)), abs(np.max(values_all)))
    def color_cell(value, lower_is_better, avg=False):
        """
        Diverging normalization centered at 0.
        Blue = improvement, Orange = degradation.
        """
        lim_for_coloring = lim_for_coloring_avg if avg else lim_for_coloring_metrics
        cmap = GOLD_PURPLE_CMAP if avg else BLUE_ORANGE_CMAP
        # symmetric limits around zero
        if np.isclose(lim_for_coloring, 0):
            norm = 0.5
        else:
            # map [-lim, 0, +lim] → [0, 0.5, 1]
            norm = (value / lim_for_coloring + 1) / 2
        # flip direction if lower is better (ECE, Brier)
        if lower_is_better:
            norm = 1 - norm

        r, g, b = [int(255 * c) for c in cmap(norm)[:3]]
        return f"\\cellcolor[RGB]{{{r},{g},{b}}} {format_value(value)}"

    # -----------------------
    # Main function
    # -----------------------
    def generate_latex_rows(df_results, model_order, estimator_order, model_name_mapping=None):
        """
        Returns a list of LaTeX table body rows (strings).
        """
        latex_rows = []

        for metric in METRICS:
            if metric != "Accuracy":
                for model in model_order:
                    display_name = model_name_mapping.get(model, model) if model_name_mapping else model
                    row_values = [display_name]
                    for setting in ["plus_none", "correct_to_none"]:
                        if "Accuracy" in df_results:
                            res = df_results["Accuracy"][model][setting]["ANLL"]
                            diff = res["difference"]
                            p_value = res["p_value"]
                            cell_text = color_cell(diff, False, metric == "Avg") if p_value < ALPHA else str(format_value(diff))
                            row_values.append(cell_text)
                        for estimator in estimator_order:
                            if estimator in df_results[metric][model][setting]:
                                res = df_results[metric][model][setting][estimator]
                                diff = res["difference"]
                                p_value = res["p_value"]
                                if np.isnan(diff):
                                    cell_text = "-"
                                else:
                                    cell_text = color_cell(diff, metric in LOWER_IS_BETTER, metric == "Avg") if p_value < ALPHA else str(format_value(diff))
                                row_values.append(cell_text)
                    
                    latex_rows.append(" & ".join(row_values) + " \\\\")
            latex_rows.append("\\hline")

        return latex_rows

    latex_body_rows = generate_latex_rows(
        df_results=df_results,
        model_order=MODELS,
        estimator_order=ESTIMATORS,
        model_name_mapping=MODEL_NAME_MAPPING,
    )

    with open(f"{args.output_folder}/none_table_all.tex", "w") as f:
        f.write("\n".join(latex_body_rows))

if __name__ == "__main__":
	args = parse_args()
	main(args)