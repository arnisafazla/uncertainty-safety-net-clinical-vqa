import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import argparse, os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the saved values in plots folder, then generate the latex table and save it in plots folder"
    )
    parser.add_argument(
        "--input-folder", "-f", required=True,
        help="Path to the main plots folder, for example outputs/GMAI-MMBench/plots/histograms"
    )
    parser.add_argument(
            "--output-folder", "-f", required=True,
            help="Path to the output plots folder, for example ./plots_and_tables"
        )

def main(args):
    # === Load and preprocess the data ===
    os.makedirs(args.output_folder, exist_ok=True)
    N = 210 # Number of samples per modality
    K = 4   # Number of shuffles per sample
    M = 8   # Number of modalities
    S = 10  # Number of repeated samplings per each shuffle
    model_name = "Qwen2.5-VL-72B-Instruct"  # or None, "Semantic Entropy", "SampleConsistency", "MeanNegativeLogLikelihood", "LabelNegativeLogLikelihood"
    per = "modality"  # or "modality", "clinical VQA task", "department", "perceptual granularity"

    file_path = f"{args.input_folder}/{model_name}_per_{per}.csv"  # Replace with actual path
    df = pd.read_csv(file_path, header=1)

    # excluded_models = ["LLaVA-OneVision-1.5-8B-Instruct", "Molmo-72B-0924", "Qwen2.5-VL-7B-Instruct", "num_samples"]
    excluded_models = []
    METHODS_MAPPING = {
        "LabelNLL": "Label NLL",
        "MeanNegativeLogLikelihood": "ANLL",
        "MaxNLL": "Max NLL",
        "SampleConsistency": "SC",
        "SemanticEntropy": "SE",
        "PRO": "PRO",
        "EigenEmbed": "EE",
        "RDS": "RDS"
    }
    df["method"] = df["method"].replace(METHODS_MAPPING)

    modality_mapping = {}
    if per == "modality":
        modalities = ['CT', 'MRI', 'Endoscopy', 'Histopathology', 'Fundus Photography',
                    'X-ray', 'Microscopy', 'Dermoscopy']
        display_names = {
            'Endoscopy': 'Endo.',
            'Histopathology': 'Histo.',
            'Fundus Photography': 'Fundus P.',
            'Microscopy': 'Micro.',
            'Dermoscopy': 'Dermo.',
            # Add other mappings if needed for wrapping long names
        }
    elif per == "department":
        modalities = ["Ophthalmology",'Dermatology','Hematology','Pulmonary','Oncology','General S.','Gastro.','Sports','Pathology','Orthopedic S.','Urology','Infectious Dis.','Cardiovascular S.','Otolaryngology','Neurosurgery','Obstetrics&Gynecology','Nephrology','Endocrinology']
        modality_mapping = {
                "Gastroenterology and Hepatology": "Gastro.",
                "Pulmonary Medicine": "Pulmonary",
                "Oncology (Medical)": "Oncology",
                "Orthopedic Surgery": "Orthopedic S.",
                "General Surgery": "General S.",
                "Hematology": "Hematology",
                "Urology": "Urology",
                "Laboratory Medicine and Pathology": "Pathology",
                "Cardiovascular Surgery": "Cardiovascular S.",
                "Ophthalmology": "Ophthalmology",
                "Dermatology": "Dermatology",
                "Otolaryngology (ENT)/Head and Neck Surgery": "Otolaryngology",
                "Nephrology and Hypertension": "Nephrology",
                "Obstetrics and Gynecology": "Obstetrics&Gynecology",
                "Neurosurgery": "Neurosurgery",
                "Sports Medicine": "Sports",
                "Endocrinology": "Endocrinology",
                "Infectious Diseases": "Infectious Dis."
            }
    elif per in ["clinical VQA task", "task"]:
        modalities = ["Diagnosis", "Severity Grading", "S. Instrument Rec.", "Microorganism Rec.", "Cell Rec.", "Abdomen Rec.", "Counting", "Bone", "Head&Neck Rec.", "Thorax Rec.", "Attribute Rec.", "Action Rec.", "Img. Quality Grading", "Blood Vessels Rec.", "S. Workflow Rec.", "Pelvic Rec.", "Muscle"]
        modality_mapping = {
                "Disease Diagnosis": "Diagnosis",
                "Surgical Instrument Recognition": "S. Instrument Rec.",
                "Cell Recognition": "Cell Rec.",
                "Image Quality Grading": "Img. Quality Grading",
                "Surgeon Action Recognition": "Action Rec.",
                "Organ Recognition - Thorax": "Thorax Rec.",
                "Blood Vessels Recognition": "Blood Vessels Rec.",
                "Bone": "Bone",
                "Organ Recognition - Head and Neck": "Head&Neck Rec.",
                "Organ Recognition - Abdomen": "Abdomen Rec.",
                "Microorganism Recognition": "Microorganism Rec.",
                "Severity Grading": "Severity Grading",
                "Attribute Recognition": "Attribute Rec.",
                "Organ Recognition - Pelvic": "Pelvic Rec.",
                "Muscle": "Muscle",
                "Surgical Workflow Recognition": "S. Workflow Rec.",
                "Counting": "Counting"
            }
        modalities = ["Diagnosis", "Recognition", "Grading"]

    elif per == "perceptual granularity":
        modalities = ["No annotation", "Contour", "Segmentation mask", "Bounding box"]
        modality_mapping = {
            "Image level": "No annotation",
            "Contour level": "Contour",
            "Mask level": "Segmentation mask", 
            "Box level": "Bounding box"
        }

    modalities = ["Overall"] + modalities  # Include Overall

    sig_file_path = f"{folder}/{model_name}_per_{per}_significance.csv"
    sig_df = pd.read_csv(sig_file_path)

    # Apply the same method name mapping to significance data
    sig_df["method"] = sig_df["method"].replace(METHODS_MAPPING)
    sig_df[per] = sig_df[per].replace(modality_mapping)

    # Create a lookup dictionary for easy access
    # Key: (method, modality, metric) -> Value: in_best_group (0 or 1)
    sig_lookup = {}
    for _, row in sig_df.iterrows():
        key = (row['method'], row[per], row['metric'])
        sig_lookup[key] = row['in_best_group']

    # Create corresponding column names for each metric
    auroc_cols = modalities
    ece_cols = [f'{spec}.1' for spec in modalities]
    brier_cols = [f'{spec}.2' for spec in modalities]

    # Convert relevant columns to numeric
    for col in auroc_cols + ece_cols + brier_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # === Prepare data for plotting ===
    plot_data = []

    # Then add individual modality data
    for i, spec in enumerate(modalities):
        auroc_col = auroc_cols[i]
        ece_col = ece_cols[i]
        brier_col = brier_cols[i]

        for _, row in df.iterrows():
            method = row['method']
            auroc_val = 1 - row[auroc_col]
            ece_val = row[ece_col]
            brier_val = row[brier_col]
            calibration_score = ((ece_val) + (brier_val)) / 2

            plot_data.append({
                'modality': spec,
                'Method': method,
                'AUROC': auroc_val,
                'Calibration': calibration_score
            })

    plot_df = pd.DataFrame(plot_data)

    # Exclude specific models
    plot_df = plot_df[~plot_df['Method'].isin(excluded_models)]

    # Color assignment function
    def get_color(method):
        method_color_map = {
            "Label NLL": "#F8AEAE",
            "ANLL": "#E8745F",
            "Max NLL": "#EA381D",
            "SC": "#56C7E9",
            "SE": "#2A47EE",
            "PRO": "#871DF8",
            "EE": "#70FE7A",
            "RDS": "#14C237"
        }

        # clean_model = model.replace('-Instruct', '').replace('-instruct', '')
        return method_color_map.get(method, "#6b7280")

    plot_df['Color'] = plot_df['Method'].apply(get_color)

    # === Create the plot ===
    if per in ["perceptual granularity", "clinical VQA task", "task"]:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig, ax = plt.subplots(figsize=(15, 5.5))
        # fig, ax = plt.subplots(figsize=(16, 5.5))

    custom_order = ["Label NLL", "ANLL", "Max NLL", "SC", "SE", "PRO", "EE", "RDS"]
    unique_methods = [m for m in custom_order if m in plot_df['Method'].unique()]

    # Update modality order to include Overall first
    n_methods = len(unique_methods)
    n_modalities = len(modalities)
    bar_width = 0.8 / n_methods
    modality_positions = np.arange(n_modalities)

    # Add alternating background for readability
    for i in range(n_modalities):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, color='lightgray', alpha=0.3)

    for i, method in enumerate(unique_methods):
        method_data = plot_df[plot_df['Method'] == method]
        x_positions = modality_positions + (i - n_methods / 2 + 0.5) * bar_width
        
        auroc_values = []
        calibration_values = []
        colors = []
        auroc_hatches = []  
        ece_hatches = []   
        
        for spec in modalities:
            row = method_data[method_data['modality'] == spec]
            if not row.empty:
                auroc = row['AUROC'].iloc[0]
                calib = row['Calibration'].iloc[0]
                color = row['Color'].iloc[0]
            else:
                auroc = 0
                calib = 0
                color = 'gray'
            
            auroc_values.append(auroc)
            calibration_values.append(calib)
            colors.append(color)
            
            # Determine hatch patterns based on significance
            # Check if this method is in best group for AUROC
            auroc_sig = sig_lookup.get((method, spec, 'AUROC'), 0)
            auroc_hatches.append('///' if auroc_sig == 1 and auroc > 0 else '')
            
            # Check if this method is in best group for ECE
            ece_sig = sig_lookup.get((method, spec, 'ECE'), 0)
            ece_hatches.append('\\\\\\' if ece_sig == 1 and calib > 0 else '')
        
        # Plot AUROC bars (top) with hatching
        for j, (x_pos, auroc_val, color, hatch) in enumerate(zip(x_positions, auroc_values, colors, auroc_hatches)):
            ax.bar(x_pos, auroc_val, bar_width,
                color=color, edgecolor='black', linewidth=0.8,
                alpha=0.8, hatch=hatch)
        
        # Plot Calibration bars (bottom) with hatching
        for j, (x_pos, calib_val, color, hatch) in enumerate(zip(x_positions, calibration_values, colors, ece_hatches)):
            ax.bar(x_pos, -calib_val, bar_width,
                color=color, edgecolor='black', linewidth=0.8,
                alpha=0.8, hatch=hatch)

    # Labels and formatting
    fontsize = 16
    ax.set_ylabel('Calibration ←    Discrimination ←', fontsize=fontsize, fontweight='bold')
    # ax.yaxis.set_label_coords(-0.04, 0.45)

    ax.yaxis.labelpad = 10   # increase until it looks good
    ax.set_xticks(modality_positions)

    fontsize = 16
    if per == "modality":
        ax.set_xticklabels(
            [r'$\mathbf{Overall}$'] + [display_names.get(s, s) for s in modalities[1:]],
            rotation=35, ha='right', fontsize=fontsize
        )
    else:
        ax.set_xticklabels(
            [r'$\mathbf{Overall}$'] + modalities[1:],  # bold only 'Overall'
            rotation=35, ha='right', fontsize=fontsize
        )
    # ax.set_xticks([])


    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Add vertical line to separate Overall from individual modalities
    # if len(unique_modalities) > 1:
    #     ax.axvline(x=0.5, color='red', linewidth=2, linestyle='-', alpha=0.7)

    # Set y-axis limits explicitly
    max_auroc = plot_df['AUROC'].max()
    max_calibration = plot_df['Calibration'].max()

    # Set limits with some padding
    y_upper = max_auroc * 1.2  # 10% padding above max AUROC
    y_lower = -max_calibration * 1.2  # 10% padding below max calibration

    ax.set_ylim(y_lower, y_upper)

    # Update y-tick labels to show absolute values
    fontsize = 18
    # if per in ["perceptual granularity", "task"] else 16
    y_ticks = ax.get_yticks()
    # Filter ticks to only show those within the data range
    y_ticks = [t for t in y_ticks if y_lower <= t <= y_upper]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{abs(t):.2f}' for t in y_ticks])
    ax.tick_params(axis='y', labelsize=fontsize)

    # Legend
    legend_elements = []
    for method in unique_methods:
        method_color = get_color(plot_df[plot_df['Method'] == method]['Method'].iloc[0])
        legend_elements.append(mpatches.Patch(color=method_color, label=method))

    fontsize = 16 if per in ["perceptual granularity", "task"] else 16
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, 1.02), loc='lower center',
            ncol=4, fontsize=fontsize, title=None)

    plt.tight_layout()
    per = per.replace("perceptual granularity", "overlay annotations")
    per = per.replace("department", "specialty")
    path = f"{args.output_folder}/{model_name}_per_{per}_significance.pdf"
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
	args = parse_args()
	main(args)