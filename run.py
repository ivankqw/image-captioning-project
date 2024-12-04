import argparse
import os
import subprocess
import sys

from nltk import download


def main():
    # Set the PYTHONPATH to the current directory
    os.environ["PYTHONPATH"] = os.getcwd()

    # Download NLTK data
    download("punkt")
    download("wordnet")

    parser = argparse.ArgumentParser(
        description="Run training or testing for a specified model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "model_1_baseline_cnn_lstm",
            "model_2_baseline_ft_cnn_lstm",
            "model_3_butd_rnn",
            "model_4_butd_lstm_att",
            "model_5_image_segmentation_lstm",
            "model_6_butd_trans_att",
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
