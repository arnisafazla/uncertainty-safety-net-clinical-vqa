import os
import argparse
import base64
from PIL import Image
from io import BytesIO
import os
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load the MedVQA/GMAI-MMBench/GMAI_mm_bench_VAL.tsv file, load and save the images, replace the image column with the saved image paths, and save the modified TSV."
    )
    parser.add_argument(
        "--filename", "-f", required=True,
        help="Name of the TSV file (e.g. GMAI_mm_bench_VAL.tsv)"
    )
    parser.add_argument(
        "--input-tsv", "-i", required=True,
        help="Path to the input TSV file (e.g. GMAI_mm_bench_VAL.tsv)"
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save the processed TSV")

    args = parser.parse_args()
    print(vars(args))
    return args

Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection

def decode_and_save(base64_str, out_path, max_size=512):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    # Resize with aspect ratio preserved
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    img.save(out_path, format="JPEG", quality=95)

def main(args):
    print(f"In main, cwd: {os.getcwd()}")
    input_tsv_path = args.input_tsv
    input_dir = os.path.dirname(input_tsv_path)
    output_dir = args.output_dir
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    df = pd.read_csv(args.input_tsv, sep="\t")

    for idx, row in tqdm(df.iterrows()):
        decode_and_save(
            row["image"],
            os.path.join(images_dir, f"{idx}.jpg"),
            max_size=512
        )

    df["image_path"] = df.apply(lambda row: f"{row.name}.jpg", axis=1)
    df.drop(columns=["image"], inplace=True)

    output_tsv_path = os.path.join(output_dir, args.filename)
    df.to_csv(output_tsv_path, sep="\t", index=False)

if __name__ == "__main__":
	args = parse_args()
	main(args)

