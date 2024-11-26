import warnings

# Suppress FutureWarning from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)
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
from metrics import (
    calculate_bleu_score,
    calculate_cider_score,
    calculate_meteor_score,
    evaluate,
)
from models.transfer_learning_model.model import DecoderRNN, EncoderCNN  # Updated import


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
    word2idx, idx2word, image_captions = build_vocabulary(caption_df, vocab_size=5000)
    captions_seqs, max_length = convert_captions_to_sequences(
        image_captions, word2idx
    )
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)
    image_names = list(image_captions.keys())
    train_images, val_images, test_images = get_splits(
        image_names, test_size=0.3
    )
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Models
    encoder = EncoderCNN(embed_size=200).to(device)
    decoder = DecoderRNN(
        embed_size=200, hidden_size=512, vocab_size=len(word2idx), num_layers=2
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<pad>"])
    # Only optimize parameters that require gradients
    params = list(filter(lambda p: p.requires_grad, encoder.parameters())) + list(
        decoder.parameters()
    )
    optimizer = optim.Adam(params, lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    val_image2captions = prepare_image2captions(val_images, captions_seqs, idx2word)
    num_epochs = 12
    train_losses = []
    val_losses = []
    val_bleu_scores = []
    val_meteor_scores = []
    val_cider_scores = []
    end_token_idx = word2idx["<end>"]

    # Timing
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        encoder.train()
        decoder.train()
        total_loss = 0
        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            lengths = torch.tensor(lengths).to(device)
            adjusted_lengths = lengths - 1
            targets = nn.utils.rnn.pack_padded_sequence(
                captions[:, 1:],
                adjusted_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )[0].to(device)
            features = encoder(images)
            outputs = decoder(features, captions)
            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs,
                adjusted_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )[0].to(device)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")
        val_loss = evaluate(
            encoder, decoder, val_loader, criterion, word2idx, device
        )
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")
        bleu_score = calculate_bleu_score(
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
        val_bleu_scores.append(bleu_score)
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
        val_meteor_scores.append(meteor)
        cider_score = calculate_cider_score(
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
        val_cider_scores.append(cider_score)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], BLEU: {bleu_score:.4f}, METEOR: {meteor:.4f}, CIDEr: {cider_score:.4f}"
        )
        # Epoch duration
        epoch_duration = time.time() - epoch_start_time
        print(
            f"Epoch [{epoch+1}/{num_epochs}] completed in {epoch_duration:.2f} seconds"
        )
        scheduler.step()
    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
    os.makedirs("models/transfer_learning_model", exist_ok=True)
    torch.save(encoder.state_dict(), "models/transfer_learning_model/encoder.pth")
    torch.save(decoder.state_dict(), "models/transfer_learning_model/decoder.pth")


if __name__ == "__main__":
    main()