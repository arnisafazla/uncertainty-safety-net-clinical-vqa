import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import argparse, os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the saved values in plots folder, then generate the latex table and save it in plots folder"
    )
    parser.add_argument(
        "--input-folder", "-f", required=True,
        help="Path to the main plots folder, for example outputs/GMAI-MMBench/plots"
    )
    parser.add_argument(
            "--output-folder", "-f", required=True,
            help="Path to the output plots folder, for example ./plots_and_tables"
        )
    
    args = parser.parse_args()
    print(vars(args))
    return args

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    MODELS = ["medgemma-4b-it","qwen_2_vl_7b_instruct", "Qwen2.5-VL-7B-Instruct", "Molmo-7B-D-0924", 
        "llava_1.6_7b", "LLaVA-OneVision-1.5-8B-Instruct"]

    UE_methods = ["LabelNLL", "ANLL", "MaxNLL", "SC", "SE", "PRO", "EE_embeddings-biomedical", "RDS_embeddings-biomedical"]

    model_mapping = {
        "medgemma-4b-it": "MedGemma-4B",
        "qwen_2_vl_7b_instruct": "Qwen2-VL-7B",
        "Qwen2.5-VL-7B-Instruct": "Qwen2.5-VL-7B",
        "Molmo-7B-D-0924": "Molmo-7B",
        "llava_med_7b": "LLaVA-Med-7B",
        "llava_1.6_7b": "LLaVA-v1.6-7B",
        "LLaVA-OneVision-1.5-8B-Instruct": "LLaVA-OV-8B",
        "medgemma-27b-it": "MedGemma-27B",
        "Qwen2.5-VL-32B-Instruct": "Qwen2.5-VL-32B",
        "llava_1.6_34b": "LLaVA-v1.6-34B",
        "Molmo-72B-0924": "Molmo-72B",
        "Qwen2.5-VL-72B-Instruct": "Qwen2.5-VL-72B",
        "gpt-4.1-mini": "GPT-4.1-Mini"
    }

    BLUE_RGB   = np.array([0, 114, 178])   / 255.0
    ORANGE_RGB = np.array([230, 159, 0])   / 255.0
    WHITE_RGB  = np.array([255, 255, 255]) / 255.0

    BLUE_CMAP = LinearSegmentedColormap.from_list(
        "blue",
        [WHITE_RGB, BLUE_RGB]    # for AUROC, higher is better
    )
    ORANGE_CMAP = LinearSegmentedColormap.from_list(
        "orange",
        [WHITE_RGB, ORANGE_RGB]  # for ECE, lower is better
    )

    def format_value(val, sig: bool):
        val = f"{val*100:.1f}"
        return "\\textbf{" + val + "}" if sig else val

    def color_cell(value, metric: str, sig: bool):
        """
        Gradient coloring with:
        - metric-specific ranges
        - lighter pastel colors
        - wider visible variation
        """
        if metric == "AUROC":
            vmin, vmax = 0.40, 0.80
            color_val = value

            norm = (color_val - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0, 1)
            norm = -0.15 + 0.7 * norm

            cmap = BLUE_CMAP

        else:
            vmin, vmax = 0.3, 0.9
            color_val = 1 - value

            norm = (color_val - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0, 1)

            # Increase separation between values
            norm = norm ** 1.3

            # Lighter + broader orange range
            norm = -0.15 + 0.8 * norm

            cmap = ORANGE_CMAP

        r, g, b = [int(255 * c) for c in cmap(norm)[:3]]

        return (
            f"\\cellcolor[RGB]{{{r},{g},{b}}} {format_value(value, sig)}"
            if sig else format_value(value, sig)
        )

    # ---------------------------------------------------------------------------------------
    sig_file_path = f"{args.input_folder}/ue_results_significance_99.csv"
    sig_df = pd.read_csv(sig_file_path)
    sig_lookup = {}
    for _, row in sig_df.iterrows():
        key = (row['method'], row['model'], row['metric'])
        sig_lookup[key] = row['in_best_group']

    df_topp = pd.read_csv(f"{args.input_folder}/ue_results_99.csv")

    # --------------------------------------------------------------------------------------------
    rows_latex = []
    for model in MODELS:
        # get the row
        row = df_topp.loc[df_topp['MODEL_NAME'] == model]
        row_str = ""
        values = []
        values.append(f"{row['avg_accuracy_majority_voting'].values[0]:.1f}")
        for ue_method in UE_methods:
            value = row[f"{ue_method}_auroc"].values[0]
            key_auroc = (ue_method, model, "AUROC")
            if sig_lookup.get(key_auroc, 0) == 1:
                auroc = color_cell(value, "AUROC", True)
            else:
                auroc = color_cell(value, "AUROC", False)
            values.extend([str(auroc)])
        for ue_method in UE_methods:
            value = row[f"{ue_method}_ece"].values[0]
            key_ece = (ue_method, model, "ECE")
            if sig_lookup.get(key_ece, 0) == 1:
                ece = color_cell(value, "ECE", True)
            else:
                ece = color_cell(value, "ECE", False)
            values.extend([str(ece)])
        row_str += " & ".join(values) + " \\\\"
        rows_latex.append(row_str)
    rows_latex = [row.replace("nan", "-") for row in rows_latex]
    output_latex = ""
    for i, model in enumerate(MODELS):
        output_latex += f"{model_mapping[model]} & {rows_latex[i].replace('nan', '-')}\n"

    with open(f"{args.output_folder}/main_table.tex", "w") as f:
        f.write(output_latex)

if __name__ == "__main__":
	args = parse_args()
	main(args)
