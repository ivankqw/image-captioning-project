import warnings

# Suppress FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import os

import pandas as pd
import torch
from PIL import Image

from data.dataset import get_transform
from data.preprocessing import (build_vocabulary,
                                convert_captions_to_sequences, get_splits,
                                prepare_image2captions)
from models.base_model.model import DecoderRNN, EncoderCNN


def main():
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
        default="models/base_model",
        help="Directory for models",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(args.data_dir, f"{args.dataset}_Dataset/Images")
    captions_file = os.path.join(args.data_dir, f"{args.dataset}_Dataset/captions.txt")
    image_dir = dataset_dir
    glove_path = "./glove/glove.6B.200d.txt"

    caption_df = pd.read_csv(captions_file).dropna().drop_duplicates()
    word2idx, idx2word, image_captions = build_vocabulary(caption_df, vocab_size=5000)
    embed_size = 200
    captions_seqs, max_length = convert_captions_to_sequences(image_captions, word2idx)
    transform = get_transform(train=False)
    image_names = list(image_captions.keys())
    _, val_images, test_images = get_splits(image_names, test_size=0.3)
    test_image2captions = prepare_image2captions(test_images, captions_seqs, idx2word)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Models
    encoder = EncoderCNN(embed_size=embed_size).to(device)
    decoder = DecoderRNN(
        embed_size=embed_size, hidden_size=512, vocab_size=len(word2idx), num_layers=2
    ).to(device)

    # Updated loading of state dicts
    encoder.load_state_dict(
        torch.load(os.path.join(args.model_dir, "encoder.pth"), map_location=device)
    )
    decoder.load_state_dict(
        torch.load(os.path.join(args.model_dir, "decoder.pth"), map_location=device)
    )

    encoder.eval()
    decoder.eval()
    end_token_idx = word2idx["<end>"]

    for img_id in test_images[:10]:
        img_path = os.path.join(image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = encoder(image)
            sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
            sampled_caption = [
                idx2word.get(word_id, "<unk>") for word_id in sampled_ids
            ]
            sampled_caption = [
                word.lower()
                for word in sampled_caption
                if word not in ["<start>", "<end>", "<pad>", "<unk>"]
            ]
        print(f"Image: {img_id}")
        print("Generated Caption:", " ".join(sampled_caption))
        print("Ground Truth Captions:")
        for ref in test_image2captions[img_id]:
            print(" ".join(ref))
        print("-" * 50)


if __name__ == "__main__":
    main()
