import os
import argparse
from dotenv import load_dotenv
from huggingface_hub.hf_api import HfFolder
from huggingface_hub import login, hf_hub_download

def main(output_dir):
	print(f"In main, cwd: {os.getcwd()}")

	# Authenticate with Hugging Face Hub
	load_dotenv()
	token = os.getenv("HF_TOKEN")
	if not token:
		raise ValueError("HF_TOKEN not found in .env or environment variables.")
	login(token=token)

	# Download file
	downloaded_path = hf_hub_download(
		repo_id="OpenGVLab/GMAI-MMBench",
		filename="GMAI_mm_bench_VAL.tsv",
		repo_type="dataset",
		local_dir=output_dir
	)

	print(f"File downloaded to: {downloaded_path}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Download file from Hugging Face Hub")
	parser.add_argument(
		"-o",
		"--output-dir",
		type=str,
		required=True,
		help="Directory where the downloaded file will be saved"
	)
	args = parser.parse_args()

	args = parser.parse_args()
	print(vars(args))
	main(args.output_dir)

