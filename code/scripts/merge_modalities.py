import os
import argparse
import os
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the MedVQA/GMAI-MMBench/GMAI_mm_bench_VAL_images_removed.tsv file, rename the modalities that start with MRI to be MRI, and save the modified TSV."
    )
    parser.add_argument(
        "--input-tsv", "-i", required=True,
        help="Path to the input TSV file (e.g. GMAI_mm_bench_VAL_images_removed.tsv)"
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save the processed TSV")

    args = parser.parse_args()
    print(vars(args))
    return args

def main(args):
    print(f"In main, cwd: {os.getcwd()}")
    input_tsv_path = args.input_tsv
    output_dir = args.output_dir
    
    df = pd.read_csv(input_tsv_path, sep="\t")

    df['modality'] = df['modality'].apply(lambda x: 'MRI' if "MRI" in x else x)

    output_tsv_path = os.path.join(output_dir, "GMAI_mm_bench_VAL_images_removed_MRI_merged.tsv")
    df.to_csv(output_tsv_path, sep="\t", index=False)

if __name__ == "__main__":
	args = parse_args()
	main(args)

