import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse
import nltk

from models.base_model import EncoderCNN, DecoderRNN
from data.dataset import FlickrDataset, collate_fn
from utils import load_glove_embeddings, captions_to_sequences, prepare_image2captions, build_vocab
from evaluate import calculate_bleu_score, calculate_meteor_score, calculate_cider_score, evaluate

nltk.download('punkt')
nltk.download('wordnet')

# Ensure reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def main():
    parser = argparse.ArgumentParser(description='Train image captioning model.')
    parser.add_argument('--dataset', type=str, required=True, choices=['Flickr8k', 'Flickr30k'])
    args = parser.parse_args()

    # Paths
    dataset_dir = f'./flickr_data/{args.dataset}_Dataset/Images'
    captions_file = f'./flickr_data/{args.dataset}_Dataset/captions.txt'
    image_dir = dataset_dir
    glove_path = "./glove/glove.6B.200d.txt" 

    # Load captions
    caption_df = pd.read_csv(captions_file).dropna().drop_duplicates()
    image_captions = caption_df.groupby("image")["caption"].apply(list).to_dict()

    # Build vocabulary
    vocab_size = 5000
    embed_size = 200
    word2idx, idx2word = build_vocab(image_captions, vocab_size=vocab_size)

    # Load GloVe embeddings based on the vocabulary created
    embedding_matrix = load_glove_embeddings(glove_path, word2idx, vocab_size, embed_size)

    # Convert captions to sequences
    captions_seqs = captions_to_sequences(image_captions, word2idx)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Split data
    image_names = list(image_captions.keys())
    train_images, temp_images = train_test_split(image_names, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    # Datasets and loaders
    train_dataset = FlickrDataset(image_dir, train_images, captions_seqs, transform=train_transform)
    val_dataset = FlickrDataset(image_dir, val_images, captions_seqs, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Models
    encoder = EncoderCNN(embed_size=200).to(device)
    decoder = DecoderRNN(embed_size=200, hidden_size=512, vocab_size=len(word2idx), embedding_matrix=embedding_matrix, num_layers=2).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Prepare image-to-captions mapping
    val_image2captions = prepare_image2captions(val_images, captions_seqs, idx2word)

    # Training loop
    num_epochs = 2
    train_losses = []
    val_losses = []
    val_bleu_scores = []
    val_meteor_scores = []
    val_cider_scores = []

    end_token_idx = word2idx['<end>']

    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        for i, (images, captions, lengths) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            lengths = torch.tensor(lengths)
            adjusted_lengths = lengths - 1

            # Targets
            targets = nn.utils.rnn.pack_padded_sequence(
                captions[:, 1:], adjusted_lengths, batch_first=True, enforce_sorted=False
            )[0]

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions)
            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, adjusted_lengths, batch_first=True, enforce_sorted=False
            )[0]

            # Loss
            loss = criterion(outputs, targets)


            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=5)
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Average loss
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

        # Evaluate
        val_loss = evaluate(encoder, decoder, val_loader, criterion, word2idx, device)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # Metrics
        bleu_score = calculate_bleu_score(
            encoder, decoder, image_dir, val_images, val_image2captions, val_transform, idx2word, device, word2idx
        )
        val_bleu_scores.append(bleu_score)

        meteor = calculate_meteor_score(
            encoder, decoder, image_dir, val_images, val_image2captions, val_transform, idx2word, device, word2idx
        )
        val_meteor_scores.append(meteor)

        cider_score = calculate_cider_score(
            encoder, decoder, image_dir, val_images, val_image2captions, val_transform, idx2word, device, word2idx
        )
        val_cider_scores.append(cider_score)

        print(f'Epoch [{epoch+1}/{num_epochs}], BLEU: {bleu_score:.4f}, METEOR: {meteor:.4f}, CIDEr: {cider_score:.4f}')

        # Step scheduler
        scheduler.step()

    # Save models in the 'models/' directory with descriptive names
    os.makedirs('models', exist_ok=True)
    torch.save(encoder.state_dict(), 'models/base_model_encoder.pth')
    torch.save(decoder.state_dict(), 'models/base_model_decoder.pth')

if __name__ == '__main__':
    main()