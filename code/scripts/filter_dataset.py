#!/usr/bin/env python3
import os, string
import argparse
import pandas as pd
import random
from itertools import chain
import warnings
from typing import Dict, List, Tuple

from code.utils.gmai_data import generate_unique_shuffles, generate_unique_shuffles_replace_option

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter, sample, upsample, and shuffle options for MedVQA TSV."
    )
    parser.add_argument(
        "--input-tsv", "-i", required=True,
        help="Path to the input TSV file (e.g. GMAI_mm_bench_VAL_images_removed.tsv)"
    )
    parser.add_argument(
        "--modalities", "-m", nargs="+", default=None,
        help="List of modalities to include (e.g. 'X-ray' 'MRI')"
    )
    parser.add_argument(
        "--n-per-modality", "-n", type=int, required=True,
        help="Number of random samples to draw per modality"
    )
    parser.add_argument(
        "--k-augment", "-k", type=int, default=1,
        help="How many times to duplicate (with shuffled options) each sample"
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save the processed TSV")

    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--new-option", type=str, default=None,
        help="If specified, replaces all answers with this new option text and then shuffles."
    )

    args = parser.parse_args()
    print(vars(args))
    return args

def main():
    args = parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.input_tsv, sep="\t")
    print(f"Unique modalities in input: {df['modality'].unique()}")
    if args.modalities:
        df_filt = df[df["modality"].isin(args.modalities)]
        modalities = args.modalities
    else:
        df_filt = df
        modalities = df["modality"].unique().tolist()

    if df_filt.empty:
        raise ValueError(f"No rows match modalities: {args.modalities}")

    sampled = []
    for mod in modalities:
        sub = df_filt[df_filt["modality"] == mod]
        n_available = len(sub)

        if n_available < args.n_per_modality:
            warnings.warn(
                f"Not enough samples for modality '{mod}': have {n_available}, want {args.n_per_modality}. Including all."
            )
            sampled.append(sub)
        else:
            sampled.append(sub.sample(args.n_per_modality, random_state=args.seed))
    df_sampled = pd.concat(sampled, ignore_index=True)

    out_rows = []
    new_option = args.new_option
    for _, row in df_sampled.iterrows():
        correct_answer = row["answer"]
        
        options = {
            letter: row[letter]
            for letter in string.ascii_uppercase[:5]
            if letter in row and row[letter] and str(row[letter]).strip() and not pd.isna(row[letter])
        }

        # Generate unique shuffles
        if new_option is not None:
            shuffles: List[Tuple[Dict[str, str], str, str]] = generate_unique_shuffles_replace_option(options, correct_answer, args.k_augment, new_option)
        else:
            shuffles: List[Tuple[Dict[str, str], str, str]] = generate_unique_shuffles(options, correct_answer, args.k_augment)

        for shuffle_idx, (shuffled_options, new_correct_letter, label) in enumerate(shuffles):
            new_row = row.copy()
            new_row["shuffled_options"] = str(shuffled_options)
            new_row["shuffled_correct_answer"] = new_correct_letter
            new_row["original_options"] = options
            new_row["original_correct_answer"] = correct_answer
            del new_row["answer"]
            new_row["shuffle_idx"] = shuffle_idx
            new_row["difficulty_label"] = label
            out_rows.append(new_row)

    out_df = pd.DataFrame(out_rows)
    out_df.rename(columns={"index": "sample_idx"}, inplace=True)
    print("HEAD: ", out_df.head())

    # Create output folder
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "filtered_data.tsv")
    
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(out_df)} rows to: {out_path}")

if __name__ == "__main__":
    main()
