import argparse
import subprocess
import sys
import os

from nltk import download


def main():
    # Set PYTHONPATH to current working directory
    os.environ["PYTHONPATH"] = os.getcwd()

    # Download NLTK data
    download("punkt")
    download("wordnet")
    download('punkt_tab')

    parser = argparse.ArgumentParser(
        description="Run training or testing for a specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "base_model",
            "transfer_learning_model",
            "transformer_model",
            "advanced_transformer_model",
        ],
        help="Specify the model to run.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Specify whether to train or test the model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Flickr8k", "Flickr30k"],
        help="Specify the dataset to use.",
    )
    args, unknown = parser.parse_known_args()
    script_path = f"models/{args.model}/{args.mode}.py"
    cmd = [sys.executable, script_path, "--dataset", args.dataset] + unknown
    subprocess.run(cmd)


if __name__ == "__main__":
    main()