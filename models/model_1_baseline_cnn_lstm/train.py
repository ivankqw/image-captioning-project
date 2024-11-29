import argparse
import os
import random
import time  # Import time for tracking epoch duration

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from data.dataset import FlickrDataset, collate_fn, get_transform
from data.preprocessing import (
    build_vocabulary,
    convert_captions_to_sequences,
    get_splits,
    prepare_image2captions
)
from model import DecoderRNN, EncoderCNN
from metrics import (
    evaluate,
    calculate_bleu_score,
    calculate_meteor_score,
    calculate_cider_score,
)


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
    
    train_losses = []
    val_losses = []
    bleu_scores = []
    meteor_scores = []
    cider_scores = []

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
    train_images, val_images, _ = get_splits(image_names, test_size=0.2)

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
    embed_size = 256
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
    optimizer = optim.Adam(params, lr=3e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Training settings
    num_epochs = 10
    total_step = len(train_loader)
    end_token_idx = word2idx["<end>"]

    # Prepare validation image IDs and references for metrics
    val_image_ids = val_images
    image2captions = prepare_image2captions(val_image_ids, captions_seqs, idx2word)

    for epoch in range(num_epochs):
        start_time = time.time()
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

            if i % 300 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{total_step}], Loss: {loss.item():.4f}"
                )

        # Calculate average training loss
        avg_train_loss = total_loss / total_step
        train_losses.append(avg_train_loss)

        # Adjust learning rate
        scheduler.step()

        # Validation
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device, vocab_size)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        # Calculate evaluation metrics
        bleu = calculate_bleu_score(
            encoder=encoder,
            decoder=decoder,
            image_dir=image_dir,
            image_ids=val_image_ids,
            image2captions=image2captions,
            transform=val_transform,
            idx2word=idx2word,
            device=device,
            word2idx=word2idx,
        )

        meteor = calculate_meteor_score(
            encoder=encoder,
            decoder=decoder,
            image_dir=image_dir,
            image_ids=val_image_ids,
            image2captions=image2captions,
            transform=val_transform,
            idx2word=idx2word,
            device=device,
            word2idx=word2idx,
        )

        cider = calculate_cider_score(
            encoder=encoder,
            decoder=decoder,
            image_dir=image_dir,
            image_ids=val_image_ids,
            image2captions=image2captions,
            transform=val_transform,
            idx2word=idx2word,
            device=device,
            word2idx=word2idx,
        )

        end_time = time.time()
        epoch_time = end_time - start_time

        print(
            f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_time:.2f} seconds."
        )
        print(
            f"BLEU Score: {bleu:.4f}, METEOR Score: {meteor:.4f}, CIDEr Score: {cider:.4f}\n"
        )
        
        val_losses.append(val_loss)
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        cider_scores.append(cider)

    # Save the models
    os.makedirs("models/model_1_baseline_cnn_lstm", exist_ok=True)
    torch.save(encoder.state_dict(), "models/model_1_baseline_cnn_lstm/encoder.pth")
    torch.save(decoder.state_dict(), "models/model_1_baseline_cnn_lstm/decoder.pth")
    
    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig('models/model_1_baseline_cnn_lstm/loss_plot.png')
    plt.close()

    # Plot evaluation metrics
    plt.figure()
    plt.plot(range(1, num_epochs + 1), bleu_scores, label='BLEU Score')
    plt.plot(range(1, num_epochs + 1), meteor_scores, label='METEOR Score')
    plt.plot(range(1, num_epochs + 1), cider_scores, label='CIDEr Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics over Epochs')
    plt.legend()
    plt.savefig('models/model_1_baseline_cnn_lstm/metrics_plot.png')
    plt.close()


if __name__ == "__main__":
    main()
