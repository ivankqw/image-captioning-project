import re
from collections import Counter
from sklearn.model_selection import train_test_split

import nltk

nltk.download("punkt")  # Ensure the Punkt tokenizer is downloaded


def tokenize(text):
    """
    Tokenizes the input text into words.
    Args:
        text (str): Input caption text.
    Returns:
        tokens (list): List of word tokens.
    """
    text = text.lower()
    tokens = nltk.tokenize.word_tokenize(text)
    return tokens


def build_vocabulary(caption_df, vocab_size=5000, tokenizing_fn=tokenize):
    """
    Builds word-to-index and index-to-word mappings based on caption data.
    Args:
        caption_df (DataFrame): DataFrame containing image filenames and their captions.
        vocab_size (int): Maximum size of the vocabulary.
    Returns:
        word2idx (dict): Mapping from words to indices.
        idx2word (dict): Mapping from indices to words.
        image_captions (dict): Mapping from image filenames to their captions.
    """
    # Group captions by image
    image_captions = caption_df.groupby("image")["caption"].apply(list).to_dict()

    # Collect all captions
    all_captions = [
        caption for captions in image_captions.values() for caption in captions
    ]

    # Tokenize all captions and count word frequencies
    all_words = [token for caption in all_captions for token in tokenizing_fn(caption)]
    word_counts = Counter(all_words)

    # Define special tokens
    special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]

    # Initialize word-to-index and index-to-word mappings
    word2idx = {token: idx for idx, token in enumerate(special_tokens)}
    idx2word = {idx: token for idx, token in enumerate(special_tokens)}

    # Add most common words to the vocabulary
    most_common = word_counts.most_common(vocab_size - len(special_tokens))
    for idx, (word, _) in enumerate(most_common, start=len(special_tokens)):
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word, image_captions


def convert_captions_to_sequences(
    image_captions, word2idx, special_token_mapping=None, tokenizing_fn=tokenize
):
    """
    Converts captions to sequences of word indices.
    Args:
        image_captions (dict): Mapping from image filenames to their captions.
        word2idx (dict): Mapping from words to indices.
    Returns:
        captions_seqs (dict): Mapping from image filenames to sequences of word indices.
        max_length (int): Maximum length of the captions.
    """
    captions_seqs = {}
    max_length = 0

    if not special_token_mapping:
        special_token_mapping = {
            "start": "<start>",
            "end": "<end>",
            "pad": "<pad>",
            "unk": "<unk>",
        }

    for img_name, captions in image_captions.items():
        seqs = []
        for caption in captions:
            # Tokenize and add start and end tokens
            tokens = (
                [special_token_mapping["start"]]
                + tokenizing_fn(caption)
                + [special_token_mapping["end"]]
            )
            # Convert tokens to indices, use <unk> for unknown words
            seq = [
                word2idx.get(token, word2idx[special_token_mapping["unk"]])
                for token in tokens
            ]
            seqs.append(seq)
            # Update maximum caption length
            max_length = max(max_length, len(seq))
        captions_seqs[img_name] = seqs

    return captions_seqs, max_length


def get_splits(image_names, test_size=0.2):
    """
    Splits the dataset into training, validation, and test sets.
    Args:
        image_names (list): List of image filenames.
        test_size (float): Proportion of the dataset to include in the test split.
    Returns:
        train_images (list): List of training image filenames.
        val_images (list): List of validation image filenames.
        test_images (list): List of test image filenames.
    """
    # Split into training and temp (validation + test) sets
    train_images, temp_images = train_test_split(
        image_names, test_size=test_size, random_state=42
    )
    # Split temp set into validation and test sets
    val_images, test_images = train_test_split(
        temp_images, test_size=0.1, random_state=42
    )
    return train_images, val_images, test_images


def prepare_image2captions(
    image_ids, captions_seqs, idx2word, special_token_mapping=None
):
    """
    Prepares a mapping from image IDs to their corresponding captions in word form.
    Args:
        image_ids (list): List of image filenames.
        captions_seqs (dict): Mapping from image filenames to sequences of word indices.
        idx2word (dict): Mapping from indices to words.
    Returns:
        image2captions (dict): Mapping from image filenames to their captions as word lists.
    """
    if not special_token_mapping:
        special_token_mapping = {
            "start": "<start>",
            "end": "<end>",
            "pad": "<pad>",
            "unk": "<unk>",
        }
    start_end_pad = (
        special_token_mapping["start"],
        special_token_mapping["end"],
        special_token_mapping["pad"],
    )
    image2captions = {}
    for img_id in image_ids:
        seqs = captions_seqs[img_id]
        captions_list = []
        for seq in seqs:
            # Convert indices back to words
            caption = [idx2word.get(idx, special_token_mapping["unk"]) for idx in seq]
            # Remove special tokens
            caption = [word.lower() for word in caption if word not in start_end_pad]
            captions_list.append(caption)
        image2captions[img_id] = captions_list
    return image2captions
