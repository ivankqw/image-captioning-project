import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from metrics import *
from model import DecoderWithTransformer, EncoderBUAttention

from data.dataset import FlickrDataset, collate_fn, get_transform
from data.preprocessing import (
    build_vocabulary,
    convert_captions_to_sequences,
    get_splits,
    prepare_image2captions,
)

# Adjust hyperparameters
embed_dim = 1024  # Dimension of word embeddings
attention_dim = 1024  # Dimension for attention network
decoder_dim = 1024  # Dimension of transformer's feed-forward network
num_layers = 6  # Number of transformer layers
num_heads = 8  # Number of attention heads
dropout = 0.1  # Dropout rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    model_name = "model_6_butd_trans_att"
    train_losses = []
    val_losses = []
    bleu_scores = []
    meteor_scores = []
    cider_scores = []

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
    print(f"Total captions loaded: {len(caption_df)}")

    # Build vocabulary
    word2idx, idx2word, image_captions = build_vocabulary(caption_df, vocab_size=5000)
    print(f"Vocabulary size: {len(word2idx)}")

    # Convert captions to sequences
    captions_seqs, max_length = convert_captions_to_sequences(
        image_captions, word2idx, return_caplens=False
    )
    print(f"Maximum caption length: {max_length}")

    # Get data transformations
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    # Split data into training and validation sets
    image_names = list(image_captions.keys())
    train_images, val_images, _ = get_splits(image_names, test_size=0.2)
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

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
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    vocab_size = len(word2idx)
    encoder = EncoderBUAttention(device=device).to(device)
    decoder = DecoderWithTransformer(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        device=device,
    ).to(device)

    # Loss and optimizer
    criterion_ce = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"]).to(device)
    params = list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(
        decoder.parameters()
    )
    optimizer = optim.Adam(params, lr=1e-4)

    # Prepare image to captions mapping for evaluation
    val_image2captions = prepare_image2captions(val_images, captions_seqs, idx2word)

    # Training settings
    num_epochs = 2  # Adjust as needed
    total_step = len(train_loader)
    end_token_idx = word2idx["<end>"]

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        encoder.train()
        decoder.train()
        total_loss = 0

        for i, (images, captions, caplens) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            caplens = torch.tensor(caplens).to(device)

            # Forward pass
            features = encoder(images)
            outputs, caps_sorted, decode_lengths = decoder(features, captions, caplens)
            targets = caps_sorted[:, 1:]  # Remove <start> token

            # Pack outputs and targets
            outputs_packed = nn.utils.rnn.pack_padded_sequence(
                outputs, decode_lengths, batch_first=True
            )
            targets_packed = nn.utils.rnn.pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            )

            # Compute loss
            loss_ce = criterion_ce(outputs_packed.data, targets_packed.data)

            # Backward and optimize
            optimizer.zero_grad()
            loss_ce.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss_ce.item()

            if i % 500 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss_ce.item():.4f}"
                )

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / total_step

        # Validation
        val_loss = evaluate(
            encoder,
            decoder,
            val_loader,
            criterion_ce,
            device,
            vocab_size,
            caplens_required=True,
        )

        # Calculate evaluation metrics
        bleu = calculate_bleu_score(
            encoder,
            decoder,
            image_dir,
            val_images,
            val_image2captions,
            val_transform,
            idx2word,
            device,
            word2idx,
            requires_wordmap=True,
        )
        meteor = calculate_meteor_score(
            encoder,
            decoder,
            image_dir,
            val_images,
            val_image2captions,
            val_transform,
            idx2word,
            device,
            word2idx,
            requires_wordmap=True,
        )
        cider = calculate_cider_score(
            encoder,
            decoder,
            image_dir,
            val_images,
            val_image2captions,
            val_transform,
            idx2word,
            device,
            word2idx,
            requires_wordmap=True,
        )

        # Print epoch summary
        epoch_duration = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"BLEU: {bleu:.4f}, "
            f"METEOR: {meteor:.4f}, "
            f"CIDEr: {cider:.4f}, "
            f"Time: {epoch_duration:.2f}s"
        )

        # Append average training loss instead of total loss
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        cider_scores.append(cider)

    # Save the models
    os.makedirs(f"models/{model_name}", exist_ok=True)
    torch.save(encoder.state_dict(), f"models/{model_name}/encoder.pth")
    torch.save(decoder.state_dict(), f"models/{model_name}/decoder.pth")

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(f"models/{model_name}/loss_plot.png")
    plt.close()

    # Plot evaluation metrics
    plt.figure()
    plt.plot(range(1, num_epochs + 1), bleu_scores, label="BLEU Score")
    plt.plot(range(1, num_epochs + 1), meteor_scores, label="METEOR Score")
    plt.plot(range(1, num_epochs + 1), cider_scores, label="CIDEr Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics over Epochs")
    plt.legend()
    plt.savefig(f"models/{model_name}/metrics_plot.png")
    plt.close()


if __name__ == "__main__":
    main()
