import argparse
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import FlickrDataset, collate_fn, get_transform
from data.preprocessing import (
    build_vocabulary,
    convert_captions_to_sequences,
    get_splits,
    prepare_image2captions,
)
from model import DecoderRNN, EncoderCNN
import nltk


def download_nltk_resources():
    """Download necessary NLTK resources if not already present."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def main():
    # Download NLTK resources if necessary
    download_nltk_resources()

    parser = argparse.ArgumentParser(description="Test image captioning model.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["Flickr8k", "Flickr30k"],
        help="Specify which dataset to use: Flickr8k or Flickr30k",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./flickr_data",
        help="Base directory for dataset",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/model_1_baseline_cnn_lstm",
        help="Directory for models",
    )
    args = parser.parse_args()

    # Paths
    dataset_dir = os.path.join(args.data_dir, f"{args.dataset}_Dataset/Images")
    captions_file = os.path.join(args.data_dir, f"{args.dataset}_Dataset/captions.txt")
    image_dir = dataset_dir

    # Load captions
    caption_df = pd.read_csv(captions_file).dropna().drop_duplicates()

    # Build vocabulary
    word2idx, idx2word, image_captions = build_vocabulary(caption_df, vocab_size=5000)

    # Convert captions to sequences
    captions_seqs, max_length = convert_captions_to_sequences(image_captions, word2idx)

    # Get data transformations
    test_transform = get_transform(train=False)

    # Split data into training, validation, and test sets
    image_names = list(image_captions.keys())
    _, _, test_images = get_splits(image_names, test_size=0.2)

    # Prepare image to captions mapping for ground truth captions
    test_image2captions = prepare_image2captions(test_images, captions_seqs, idx2word)

    # Create test dataset and data loader
    test_dataset = FlickrDataset(
        image_dir, test_images, captions_seqs, transform=test_transform, mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    embed_size = 256
    hidden_size = 512
    vocab_size = len(word2idx)

    encoder = EncoderCNN(embed_size=embed_size).to(device)
    decoder = DecoderRNN(
        embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size
    ).to(device)

    # Load trained models
    encoder_path = os.path.join(args.model_dir, "encoder.pth")
    decoder_path = os.path.join(args.model_dir, "decoder.pth")

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
        print(f"Model files not found in {args.model_dir}. Please check the path.")
        return

    encoder.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True)
    )
    decoder.load_state_dict(
        torch.load(decoder_path, map_location=device, weights_only=True)
    )

    encoder.eval()
    decoder.eval()
    end_token_idx = word2idx.get("<end>", None)

    if end_token_idx is None:
        print("The vocabulary does not contain an <end> token.")
        return

    # Generate captions on test images
    for i, (images, captions, image_ids) in enumerate(test_loader):
        if i >= 10:
            break  # Stop after processing 10 images

        images = images.to(device)
        with torch.no_grad():
            features = encoder(images)
            sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)

        # Convert word IDs to words
        sampled_caption = [idx2word.get(word_id, "<unk>") for word_id in sampled_ids]
        generated_caption = " ".join(sampled_caption)

        # Get ground truth captions
        image_name = image_ids[0]
        gt_captions = test_image2captions.get(image_name, [])

        if not gt_captions:
            print(f"Image ID: {image_name}")
            print("Generated Caption:", generated_caption)
            print("Ground Truth Captions: None")
            print("------------------------------------")
            continue

        print(f"Image ID: {image_name}")
        print(f"Generated Caption: {generated_caption}")
        print("Ground Truth Captions:")
        for gt_caption in gt_captions:
            print(f"- {gt_caption}")
        print("------------------------------------")


if __name__ == "__main__":
    main()
