import re
import numpy as np
from collections import Counter

def tokenize(text):
    """
    Tokenizes a given text into a list of words.
    Lowercases the text and extracts alphanumeric words.
    """
    text = text.lower()
    tokens = re.findall(r'\w+', text)
    return tokens

def build_vocab(image_captions, vocab_size=5000):
    """
    Builds the vocabulary from image captions.
    """
    # Collect all captions into a list
    all_captions = []
    for captions in image_captions.values():
        all_captions.extend(captions)

    # Tokenize all captions and build a list of all words
    all_words = []
    for caption in all_captions:
        tokens = tokenize(caption)
        all_words.extend(tokens)

    # Count the frequency of each word in the dataset
    word_counts = Counter(all_words)

    # Define special tokens
    special_tokens = ['<pad>', '<start>', '<end>', '<unk>']

    # Create mappings from words to indices and vice versa
    word2idx = {token: idx for idx, token in enumerate(special_tokens)}
    idx2word = {idx: token for idx, token in enumerate(special_tokens)}

    # Get the most common words up to the vocab_size limit
    most_common = word_counts.most_common(vocab_size - len(special_tokens))

    # Add the most common words to the vocabulary
    for idx, (word, _) in enumerate(most_common, start=len(special_tokens)):
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word

def load_glove_embeddings(glove_path, word2idx, vocab_size, embedding_dim=200):
    """Loads GloVe embeddings and builds an embedding matrix for the given vocabulary."""
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # Initialize embedding matrix with zeros for OOV (out-of-vocabulary) words
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Fill embedding matrix for words in vocabulary (word2idx)
    for word, idx in word2idx.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        # OOV words will have embedding vectors of zeros (already handled by np.zeros)

    return embedding_matrix

def captions_to_sequences(captions, word2idx):
    """Converts captions to sequences of word indices."""
    sequences = {}
    for img_name, captions_list in captions.items():
        sequences[img_name] = [
            [word2idx.get(word, word2idx["<unk>"]) for word in ["<start>"] + tokenize(caption) + ["<end>"]]
            for caption in captions_list
        ]
    return sequences

def prepare_image2captions(image_ids, captions_seqs, idx2word):
    """
    Prepares a mapping from image IDs to lists of reference captions.
    """
    image2captions = {}
    for img_id in image_ids:
        seqs = captions_seqs[img_id]
        captions_list = []
        for seq in seqs:
            # Convert word indices to words
            caption = [idx2word.get(idx, '<unk>') for idx in seq]
            # Remove special tokens and lowercase the words
            caption = [word.lower() for word in caption if word not in ['<start>', '<end>', '<pad>']]
            captions_list.append(caption)
        image2captions[img_id] = captions_list
    return image2captions
