import os

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image

def evaluate(encoder, decoder, data_loader, criterion, device, vocab_size):
    """
    Evaluate the model on the validation set.

    Args:
        encoder (EncoderCNN): The encoder model.
        decoder (DecoderRNN): The decoder model.
        data_loader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.
        vocab_size (int): Size of the vocabulary.

    Returns:
        float: Average validation loss.
    """
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for images, captions, _ in data_loader:
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            global_features, object_features = encoder(images)
            outputs = decoder(global_features, object_features, captions)  # Shape: (batch_size, seq_len -1, vocab_size)

            # Targets are the next words in the captions
            targets = captions[:, 1:]  # Shape: (batch_size, seq_len -1)

            # Reshape outputs and targets for loss computation
            outputs = outputs.reshape(-1, vocab_size)  # Shape: (batch_size * (seq_len -1), vocab_size)
            targets = targets.reshape(-1)              # Shape: (batch_size * (seq_len -1))

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_batches += 1

    average_loss = total_loss / total_batches
    return average_loss

# Function to calculate BLEU score for generated captions
def calculate_bleu_score(
    encoder,
    decoder,
    image_dir,
    image_ids,
    image2captions,
    transform,
    idx2word,
    device,
    word2idx,
):
    """
    Calculate BLEU score for the generated captions.
    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        image_dir: Directory containing images.
        image_ids: List of image IDs.
        image2captions: Dictionary mapping image IDs to reference captions.
        transform: Preprocessing transformation for images.
        idx2word: Mapping from word indices to words.
        device: Computation device (CPU or GPU).
        word2idx: Mapping from words to word indices.
    Returns:
        bleu_score: Corpus BLEU score for generated captions.
    """
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    references = []  # List to store reference captions
    hypotheses = []  # List to store generated captions
    smoothie = SmoothingFunction().method4  # Smoothing function for BLEU score

    with torch.no_grad():
        for img_id in image_ids:
            # Load and preprocess image
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # Generate caption
            global_features, object_features = encoder(image)
            start_token_idx = word2idx["<start>"]
            end_token_idx = word2idx["<end>"]
            sampled_ids = decoder.sample(global_features, object_features, start_token_idx=start_token_idx, end_token_idx=end_token_idx)
            sampled_caption = [
                idx2word.get(word_id, "<unk>") for word_id in sampled_ids
            ]

            # Prepare hypothesis (generated caption tokens)
            hypothesis = [
                word.lower()
                for word in sampled_caption
                if word not in ["<start>", "<end>", "<pad>", "<unk>"]
            ]
            hypotheses.append(hypothesis)

            # Prepare references (list of lists of tokens)
            ref_captions = image2captions[img_id]
            refs = [
                [
                    word.lower()
                    for word in ref
                    if word not in ["<start>", "<end>", "<pad>", "<unk>"]
                ]
                for ref in ref_captions
            ]
            references.append(refs)

    # Compute corpus BLEU score
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return bleu_score

# Function to calculate METEOR score for generated captions
def calculate_meteor_score(
    encoder,
    decoder,
    image_dir,
    image_ids,
    image2captions,
    transform,
    idx2word,
    device,
    word2idx,
):
    """
    Calculate METEOR score for the generated captions.
    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        image_dir: Directory containing images.
        image_ids: List of image IDs.
        image2captions: Dictionary mapping image IDs to reference captions.
        transform: Preprocessing transformation for images.
        idx2word: Mapping from word indices to words.
        device: Computation device (CPU or GPU).
        word2idx: Mapping from words to word indices.
    Returns:
        average_meteor: Average METEOR score.
    """
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    meteor_scores = []  # List to store METEOR scores

    with torch.no_grad():
        for img_id in image_ids:
            # Load and preprocess image
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # Generate caption
            global_features, object_features = encoder(image)
            start_token_idx = word2idx["<start>"]
            end_token_idx = word2idx["<end>"]
            sampled_ids = decoder.sample(global_features, object_features, start_token_idx=start_token_idx, end_token_idx=end_token_idx)
            sampled_caption = [
                idx2word.get(word_id, "<unk>") for word_id in sampled_ids
            ]

            # Prepare hypothesis (generated caption tokens)
            hypothesis = [
                word.lower()
                for word in sampled_caption
                if word not in ["<start>", "<end>", "<pad>", "<unk>"]
            ]

            # Prepare references (list of lists of tokens)
            references = [
                [
                    word.lower()
                    for word in ref
                    if word not in ["<start>", "<end>", "<pad>", "<unk>"]
                ]
                for ref in image2captions[img_id]
            ]

            # Calculate METEOR score for the current image
            score = meteor_score(references, hypothesis)
            meteor_scores.append(score)

    # Compute average METEOR score
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    return average_meteor

# Function to calculate CIDEr score for generated captions
def calculate_cider_score(
    encoder,
    decoder,
    image_dir,
    image_ids,
    image2captions,
    transform,
    idx2word,
    device,
    word2idx,
):
    """
    Calculate CIDEr score for the generated captions.
    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        image_dir: Directory containing images.
        image_ids: List of image IDs.
        image2captions: Dictionary mapping image IDs to reference captions.
        transform: Preprocessing transformation for images.
        idx2word: Mapping from word indices to words.
        device: Computation device (CPU or GPU).
        word2idx: Mapping from words to word indices.
    Returns:
        cider_score: CIDEr score for generated captions.
    """
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    gts = {}  # Ground truth captions
    res = {}  # Generated captions
    tokenizer = PTBTokenizer()  # Tokenizer for captions

    with torch.no_grad():
        for img_id in image_ids:
            # Load and preprocess image
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # Generate caption
            global_features, object_features = encoder(image)
            start_token_idx = word2idx["<start>"]
            end_token_idx = word2idx["<end>"]
            sampled_ids = decoder.sample(global_features, object_features, start_token_idx=start_token_idx, end_token_idx=end_token_idx)
            sampled_caption = [
                idx2word.get(word_id, "<unk>") for word_id in sampled_ids
            ]
            
            # Prepare generated caption
            sampled_caption = [
                word.lower()
                for word in sampled_caption
                if word not in ["<start>", "<end>", "<pad>", "<unk>"]
            ]
            sampled_caption_str = " ".join(sampled_caption)

            # Prepare references
            references = [
                " ".join(
                    [
                        word.lower()
                        for word in ref
                        if word not in ["<start>", "<end>", "<pad>", "<unk>"]
                    ]
                )
                for ref in image2captions[img_id]
            ]

            # Update dictionaries with tokenized captions
            gts[img_id] = [{'caption': ref} for ref in references]
            res[img_id] = [{'caption': sampled_caption_str}]

    # Tokenize captions
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Compute CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    return cider_score
