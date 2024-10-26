import os
from PIL import Image
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize 

def evaluate(encoder, decoder, data_loader, criterion, word2idx, device):
    """
    Evaluate the model on a validation set and compute the average loss.
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
            adjusted_lengths = lengths - 1

            targets = nn.utils.rnn.pack_padded_sequence(
                captions[:, 1:], adjusted_lengths, batch_first=True, enforce_sorted=False
            )[0]

            features = encoder(images)
            outputs = decoder(features, captions)
            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, adjusted_lengths, batch_first=True, enforce_sorted=False
            )[0]

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_samples += 1

    average_loss = total_loss / total_samples
    return average_loss

def calculate_bleu_score(encoder, decoder, image_dir, image_ids, image2captions, transform, idx2word, device, word2idx):
    """
    Calculate the BLEU score for the generated captions against the reference captions.
    """
    encoder.eval()
    decoder.eval()
    references = []
    hypotheses = []
    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for img_id in image_ids:
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            features = encoder(image)
            end_token_idx = word2idx['<end>']
            sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
            sampled_caption = [idx2word.get(word_id, '<unk>') for word_id in sampled_ids]
            sampled_caption = [word.lower() for word in sampled_caption if word not in ['<start>', '<end>', '<pad>']]

            hypotheses.append(sampled_caption)
            ref_captions = image2captions[img_id]
            references.append(ref_captions)

    # Compute BLEU score
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    return bleu_score

def calculate_meteor_score(encoder, decoder, image_dir, image_ids, image2captions, transform, idx2word, device, word2idx):
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
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            # Generate caption
            features = encoder(image)
            end_token_idx = word2idx['<end>']
            sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
            
            # Convert word indices to words
            sampled_caption = [idx2word.get(word_id, '<unk>') for word_id in sampled_ids]
            
            # Remove special tokens and lowercase the words
            hypothesis = ' '.join([word.lower() for word in sampled_caption if word not in ['<start>', '<end>', '<pad>']])
            
            # Tokenize the hypothesis for meteor_score
            tokenized_hypothesis = word_tokenize(hypothesis)

            # Prepare references (list of tokenized references)
            references = [' '.join(ref) for ref in image2captions[img_id]]
            tokenized_references = [word_tokenize(ref) for ref in references]

            # Compute METEOR score
            score = meteor_score(tokenized_references, tokenized_hypothesis)  # Tokenized inputs
            meteor_scores.append(score)

    average_meteor = sum(meteor_scores) / len(meteor_scores)
    return average_meteor

def calculate_cider_score(encoder, decoder, image_dir, image_ids, image2captions, transform, idx2word, device, word2idx):
    """
    Calculate the CIDEr score for the generated captions against the reference captions.
    """
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    encoder.eval()
    decoder.eval()
    gts = {}
    res = {}
    tokenizer = PTBTokenizer()

    with torch.no_grad():
        for img_id in image_ids:
            # Load and preprocess the image
            img_path = os.path.join(image_dir, img_id)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            # Generate caption
            features = encoder(image)
            end_token_idx = word2idx['<end>']
            sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)

            # Convert word indices to words
            sampled_caption = [idx2word.get(word_id, '<unk>') for word_id in sampled_ids]

            # Remove special tokens and lowercase the words
            sampled_caption = [word.lower() for word in sampled_caption if word not in ['<start>', '<end>', '<pad>']]
            sampled_caption_str = ' '.join(sampled_caption)

            # Prepare reference captions
            ref_captions = [
                {'caption': ' '.join([word.lower() for word in ref if word not in ['<start>', '<end>', '<pad>']])}
                for ref in image2captions[img_id]
            ]

            # Assign to gts and res (formatted for CIDEr scorer)
            gts[img_id] = [{'caption': ' '.join(ref)} for ref in image2captions[img_id]]
            res[img_id] = [{'caption': sampled_caption_str}]

    # Tokenize captions (gts = reference captions, res = generated captions)
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # Compute CIDEr score
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)

    return cider_score
