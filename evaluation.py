# Import necessary libraries
import os
import torch
import torch.nn as nn
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from cfg import device


# Function to calculate BLEU score on a set of images
def calculate_bleu_score(
    encoder,
    decoder,
    image_dir,
    image_ids,
    generate_caption_ids,
    image2captions,
    transform,
    word2idx,
    idx2word,
):
    """
    Calculate the BLEU score for the generated captions against the reference captions.
    """
    encoder.eval()
    decoder.eval()
    references = []
    hypotheses = []
    smoothie = SmoothingFunction().method4  # Smoothing function for BLEU

    with torch.no_grad():
        for img_id in image_ids:
            # Load and preprocess the image
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            # Generate caption
            features = encoder(image)
            sampled_ids = generate_caption_ids(
                decoder, features, word2idx, max_length=100
            )
            # Convert word indices to words
            sampled_caption = [
                idx2word.get(word_id, "<unk>") for word_id in sampled_ids
            ]
            # Remove special tokens and lower case
            sampled_caption = [
                word.lower()
                for word in sampled_caption
                if word not in ["<start>", "<end>", "<pad>"]
            ]

            # Append hypothesis and references
            hypotheses.append(sampled_caption)
            # References need to be lowercased as well
            ref_captions = [
                [word.lower() for word in ref] for ref in image2captions[img_id]
            ]
            # ref_captions = image2captions[img_id]
            references.append(ref_captions)
    # Compute BLEU score
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return bleu_score


# Function to calculate METEOR score on a set of images
def calculate_meteor_score(
    encoder,
    decoder,
    image_dir,
    image_ids,
    generate_caption_ids,
    image2captions,
    transform,
    word2idx,
    idx2word,
):
    """
    Calculate the METEOR score for the generated captions against the reference captions.
    """
    encoder.eval()
    decoder.eval()
    meteor_scores = []

    with torch.no_grad():
        for img_id in image_ids:
            # Load and preprocess the image
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)
            # Generate caption
            features = encoder(image)
            sampled_ids = generate_caption_ids(
                decoder, features, word2idx, max_length=100
            )

            # Convert word indices to words
            sampled_caption = [
                idx2word.get(word_id, "<unk>") for word_id in sampled_ids
            ]

            # Remove special tokens
            sampled_caption = [
                word
                for word in sampled_caption
                if word not in ["<start>", "<end>", "<pad>"]
            ]

            # Hypothesis is a list of tokens (do not join into a string)
            hypothesis = sampled_caption  # List of tokens

            # References are lists of tokens (do not join into strings)
            references = image2captions[img_id]  # List of lists of tokens

            # Compute METEOR score
            score = meteor_score(references, hypothesis)
            meteor_scores.append(score)

    average_meteor = sum(meteor_scores) / len(meteor_scores)
    return average_meteor


# Function to prepare image to captions mappings for evaluation
def prepare_image2captions(image_ids, captions_seqs, idx2word):
    """
    Prepare a dictionary mapping image IDs to lists of reference captions (each as a list of tokens).
    """
    image2captions = {}
    for img_id in image_ids:
        seqs = captions_seqs[img_id]
        captions_list = []
        for seq in seqs:
            # Convert word indices to words
            caption = [idx2word.get(idx, "<unk>") for idx in seq]
            # Remove special tokens
            caption = [
                word for word in caption if word not in ["<start>", "<end>", "<pad>"]
            ]
            captions_list.append(caption)
        image2captions[img_id] = captions_list
    return image2captions


# Evaluation Function to compute average loss on a data loader
def evaluate(encoder, decoder, data_loader, criterion):
    """
    Evaluate the model on a given data loader and compute the average loss.
    """
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for images, captions, lengths in data_loader:
            images = images.to(device)
            captions = captions.to(device)
            lengths = torch.tensor(lengths)
            adjusted_lengths = (
                lengths - 1
            )  # Adjust lengths for excluding the last token

            # Prepare targets by excluding the <start> token
            targets = nn.utils.rnn.pack_padded_sequence(
                captions[:, 1:],
                adjusted_lengths,
                batch_first=True,
                enforce_sorted=False,
            )[0]

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions)
            # Pack the outputs
            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, adjusted_lengths, batch_first=True, enforce_sorted=False
            )[0]

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_samples += 1

    average_loss = total_loss / total_samples
    return average_loss
