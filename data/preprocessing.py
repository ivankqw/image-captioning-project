import re
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split

def tokenize(text):
    """
    Tokenize the input text into words.
    """
    text = text.lower()
    tokens = re.findall(r"\w+", text)
    return tokens

def build_vocabulary(caption_df, vocab_size=10000):
    """
    Build word-to-index and index-to-word mappings based on caption data.
    Args:
        caption_df: DataFrame containing image filenames and their captions.
        vocab_size: Maximum size of the vocabulary.
    Returns:
        word2idx: Dictionary mapping words to indices.
        idx2word: Dictionary mapping indices to words.
        image_captions: Dictionary mapping image filenames to their captions.
    """
    image_captions = caption_df.groupby("image")["caption"].apply(list).to_dict()
    all_captions = [
        caption for captions in image_captions.values() for caption in captions
    ]
    all_words = [token for caption in all_captions for token in tokenize(caption)]
    word_counts = Counter(all_words)
    special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
    word2idx = {token: idx for idx, token in enumerate(special_tokens)}
    idx2word = {idx: token for idx, token in enumerate(special_tokens)}
    most_common = word_counts.most_common(vocab_size - len(special_tokens))
    for idx, (word, _) in enumerate(most_common, start=len(special_tokens)):
        word2idx[word] = idx
        idx2word[idx] = word
    return word2idx, idx2word, image_captions

def convert_captions_to_sequences(image_captions, word2idx):
    """
    Convert captions to sequences of word indices.
    Args:
        image_captions: Dictionary mapping image filenames to their captions.
        word2idx: Dictionary mapping words to indices.
    Returns:
        captions_seqs: Dictionary mapping image filenames to sequences of word indices.
        max_length: Maximum length of the captions.
    """
    captions_seqs = {}
    max_length = 0
    for img_name, captions in image_captions.items():
        seqs = []
        for caption in captions:
            tokens = ["<start>"] + tokenize(caption) + ["<end>"]
            seq = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
            seqs.append(seq)
            max_length = max(max_length, len(seq))
        captions_seqs[img_name] = seqs
    return captions_seqs, max_length

def get_splits(image_names, test_size=0.3):
    """
    Split the dataset into training, validation, and test sets.
    Args:
        image_names: List of image filenames.
        test_size: Proportion of the dataset to include in the test split.
    Returns:
        train_images: List of training image filenames.
        val_images: List of validation image filenames.
        test_images: List of test image filenames.
    """
    train_images, temp_images = train_test_split(
        image_names, test_size=test_size, random_state=42
    )
    val_images, test_images = train_test_split(
        temp_images, test_size=0.5, random_state=42
    )
    return train_images, val_images, test_images

def prepare_image2captions(image_ids, captions_seqs, idx2word):
    """
    Prepare a mapping from image IDs to their corresponding captions in word form.
    Args:
        image_ids: List of image filenames.
        captions_seqs: Dictionary mapping image filenames to sequences of word indices.
        idx2word: Dictionary mapping indices to words.
    Returns:
        image2captions: Dictionary mapping image filenames to their captions in words.
    """
    image2captions = {}
    for img_id in image_ids:
        seqs = captions_seqs[img_id]
        captions_list = []
        for seq in seqs:
            caption = [idx2word.get(idx, "<unk>") for idx in seq]
            caption = [
                word.lower()
                for word in caption
                if word not in ["<start>", "<end>", "<pad>"]
            ]
            captions_list.append(caption)
        image2captions[img_id] = captions_list
    return image2captions
