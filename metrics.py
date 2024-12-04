import os

import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image


# Evaluate function: Computes validation loss on a given dataset
def evaluate(
    encoder,
    decoder,
    data_loader,
    criterion,
    device,
    vocab_size,
    caplens_required=False,
    has_extra_timestep=True,
):
    """
    Evaluate the model on the validation set.
    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        data_loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Computation device (CPU or GPU).
        vocab_size: Size of the vocabulary.
        caplens: List of caption lengths.
    Returns:
        average_loss: Average validation loss.
    """
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    total_loss = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, captions, caplens in data_loader:
            # Move data to the computation device
            images = images.to(device)
            captions = captions.to(device)
            caplens = torch.tensor(caplens).to(device)

            # Forward pass through encoder and decoder
            features = encoder(images)
            if caplens_required:

                scores, caps_sorted, decode_lengths = decoder(
                    features, captions, caplens
                )
                # since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = caps_sorted[:, 1:]

                # remove timesteps we didn't decode at, or are pads
                scores_packed_seq = nn.utils.rnn.pack_padded_sequence(
                    scores, decode_lengths, batch_first=True
                )
                targets_packed_seq = nn.utils.rnn.pack_padded_sequence(
                    targets, decode_lengths, batch_first=True
                )
                outputs = scores_packed_seq.data
                targets = targets_packed_seq.data
            else:

                outputs = decoder(features, captions)
                if has_extra_timestep:
                    # Exclude the first time step from outputs and targets
                    outputs = outputs[
                        :, 1:, :
                    ]  # Ensure outputs and targets have the same length

                targets = captions[
                    :, 1:
                ]  # Exclude the first <start> token from targets

                # Reshape outputs and targets for loss computation
                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_samples += 1

    # Calculate average loss
    average_loss = total_loss / total_samples
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
    requires_wordmap=False,
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
        requires_wordmap: whether wordmap is required during sampling
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
            features = encoder(image)
            end_token_idx = word2idx["<end>"]
            if requires_wordmap:
                sampled_ids = decoder.sample(
                    features=features, end_token_idx=end_token_idx, word_map=word2idx
                )
            else:
                sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
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
    requires_wordmap=False,
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
        requires_wordmap: whether wordmap is required during sampling
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
            features = encoder(image)
            end_token_idx = word2idx["<end>"]
            if requires_wordmap:
                sampled_ids = decoder.sample(
                    features=features, end_token_idx=end_token_idx, word_map=word2idx
                )
            else:
                sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
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
    requires_wordmap=False,
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
        requires_wordmap: whether wordmap is required during sampling
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
            features = encoder(image)
            end_token_idx = word2idx["<end>"]
            if requires_wordmap:
                sampled_ids = decoder.sample(
                    features=features, end_token_idx=end_token_idx, word_map=word2idx
                )
            else:
                sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
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
            gts[img_id] = [{"caption": ref} for ref in references]
            res[img_id] = [{"caption": sampled_caption_str}]

    # Tokenize captions
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Compute CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    return cider_score
