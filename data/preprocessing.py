import re
from collections import Counter

import nltk
nltk.download('punkt')  # Ensure the Punkt tokenizer is downloaded

def clean_caption(caption):
    """
    Cleans the input caption by removing numbers, special characters,
    and excessive whitespace. Also standardizes punctuation.
    """
    caption = caption.lower()
    caption = re.sub(r'\d+', '', caption)  # Remove numbers
    caption = re.sub(r'[^\w\s.,]', '', caption)  # Remove non-alphanumeric chars except basic punctuation
    caption = re.sub(r'\s+', ' ', caption).strip()  # Remove extra spaces
    return caption

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

def build_vocabulary(caption_df, vocab_size=8000):
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
    image_captions = caption_df.groupby("image")["caption"].apply(list).to_dict()
    all_captions = [caption for captions in image_captions.values() for caption in captions]
    all_words = [token for caption in all_captions for token in tokenize(caption)]
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

def convert_captions_to_sequences(image_captions, word2idx):
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

    for img_name, captions in image_captions.items():
        seqs = []
        for caption in captions:
            tokens = ["<start>"] + tokenize(caption) + ["<end>"]
            seq = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
            seqs.append(seq)
            max_length = max(max_length, len(seq))
        captions_seqs[img_name] = seqs

    return captions_seqs, max_length

def prepare_image2captions(image_ids, captions_seqs, idx2word):
    """
    Prepares a mapping from image IDs to their corresponding captions in word form.
    Args:
        image_ids (list): List of image filenames.
        captions_seqs (dict): Mapping from image filenames to sequences of word indices.
        idx2word (dict): Mapping from indices to words.
    Returns:
        image2captions (dict): Mapping from image filenames to their captions as word lists.
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
