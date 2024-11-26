import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

# Ensure reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from data.dataset import FlickrDataset, collate_fn, get_transform
from data.preprocessing import (
    build_vocabulary,
    convert_captions_to_sequences,
    get_splits,
    prepare_image2captions,
)
from model import DecoderRNN, EncoderCNN  # Import the model
from metrics import evaluate

def main():
    parser = argparse.ArgumentParser(description="Train image captioning model.")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["Flickr8k", "Flickr30k"]
    )
    args = parser.parse_args()

    # Paths
    dataset_dir = f"./flickr_data/{args.dataset}_Dataset/Images"
    captions_file = f"./flickr_data/{args.dataset}_Dataset/captions.txt"
    image_dir = dataset_dir

    # Load captions
    caption_df = pd.read_csv(captions_file).dropna().drop_duplicates()

    # Build vocabulary
    word2idx, idx2word, image_captions = build_vocabulary(caption_df, vocab_size=5000)

    # Convert captions to sequences
    captions_seqs, max_length = convert_captions_to_sequences(
        image_captions, word2idx
    )

    # Get data transformations
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    # Split data into training and validation sets
    image_names = list(image_captions.keys())
    train_images, val_images, _ = get_splits(image_names, test_size=0.3)

    # Create datasets and data loaders
    train_dataset = FlickrDataset(
        image_dir, train_images, captions_seqs, transform=train_transform
    )
    val_dataset = FlickrDataset(
        image_dir, val_images, captions_seqs, transform=val_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    embed_size = 200
    hidden_size = 512
    vocab_size = len(word2idx)
    encoder = EncoderCNN(embed_size=embed_size).to(device)
    decoder = DecoderRNN(
        embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    params = list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(
        decoder.parameters()
    )
    optimizer = optim.Adam(params, lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Training settings
    num_epochs = 4
    total_step = len(train_loader)
    end_token_idx = word2idx["<end>"]

    # Training loop
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions)

            # Prepare targets
            targets = captions[:, 1:]  # Exclude the first <start> token

            # Exclude the first time step from outputs
            outputs = outputs[:, 1:, :]  # Now outputs and targets have the same sequence length

            # Reshape for loss computation
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}"
                )

        # Adjust learning rate
        scheduler.step()

        # Validation
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device, vocab_size)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/total_step:.4f}, Validation Loss: {val_loss:.4f}"
        )

    # Save the models
    os.makedirs("models/model_1_baseline_cnn_lstm", exist_ok=True)
    torch.save(encoder.state_dict(), "models/model_1_baseline_cnn_lstm/encoder.pth")
    torch.save(decoder.state_dict(), "models/model_1_baseline_cnn_lstm/decoder.pth")


if __name__ == "__main__":
    main()
