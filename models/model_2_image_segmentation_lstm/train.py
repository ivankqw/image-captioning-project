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
from metrics import *

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
    print(f"Total captions loaded: {len(caption_df)}")

    # Build vocabulary
    word2idx, idx2word, image_captions = build_vocabulary(caption_df, vocab_size=10000)
    print(f"Vocabulary size: {len(word2idx)}")

    # Convert captions to sequences
    captions_seqs, max_length = convert_captions_to_sequences(image_captions, word2idx)
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

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    embed_size = 256
    hidden_size = 512
    vocab_size = len(word2idx)
    top_k = 5  # Number of objects to consider

    # Initialize encoder and decoder
    encoder = EncoderCNN(embed_size=embed_size, device=device, top_k=top_k).to(device)
    decoder = DecoderRNN(
        input_size=embed_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        dropout=0.5
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=5e-4, weight_decay=3e-4)

    # Initialize the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # We want to minimize the validation loss
        factor=0.5,           # Factor by which the learning rate will be reduced
        patience=2,           # Number of epochs with no improvement after which learning rate will be reduced
    )
    
    # Prepare image to captions mapping for evaluation
    val_image2captions = prepare_image2captions(val_images, captions_seqs, idx2word)

    # Training settings
    num_epochs = 10
    total_step = len(train_loader)
    end_token_idx = word2idx["<end>"]

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        encoder.train()
        decoder.train()
        total_loss = 0

        for i, (images, captions, _) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            global_features, object_features = encoder(images)
            outputs = decoder(global_features, object_features, captions)  # No slicing

            targets = captions[:, 1:]     # Shape: (batch_size, seq_len -1)

            # Reshape outputs and targets for loss computation
            outputs = outputs.reshape(-1, vocab_size)  # Shape: (batch_size * (seq_len -1), vocab_size)
            targets = targets.reshape(-1)              # Shape: (batch_size * (seq_len -1))

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

        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / total_step

        # Validation
        val_loss = evaluate(encoder, decoder, val_loader, criterion, device, vocab_size)

        # Step the scheduler with the validation loss
        scheduler.step(val_loss)

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

        # Save metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        cider_scores.append(cider)

    # Save the models
    os.makedirs("models/model_2_image_segmentation_lstm", exist_ok=True)
    torch.save(encoder.state_dict(), "models/model_2_image_segmentation_lstm/encoder.pth")
    torch.save(decoder.state_dict(), "models/model_2_image_segmentation_lstm/decoder.pth")
    print("Models saved successfully.")
    
    # Plot training and validation loss
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.savefig('models/model_2_image_segmentation_lstm/loss_plot.png')
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
    plt.savefig('models/model_2_image_segmentation_lstm/metrics_plot.png')
    plt.close()

if __name__ == "__main__":
    main()
