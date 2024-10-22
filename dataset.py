import os
from collections import Counter, defaultdict
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from cfg import device
from typing import List, Dict
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd


# Group captions by image
def group_captions_by_image(caption_df):
    # Grouping Captions by Image
    # Create a dictionary mapping each image name to its list of captions
    image_captions = defaultdict(list)
    for idx, row in caption_df.iterrows():
        image_captions[row["image"]].append(row["caption"])
    return image_captions


def tokenize(text):
    # Function to tokenize text into words
    # Convert text to lowercase
    text = text.lower()
    # Use regular expressions to extract words (alphanumeric characters)
    tokens = re.findall(r"\w+", text)
    return tokens


def build_vocabulary(caption_path):
    # Load the captions from the file
    caption_df = pd.read_csv(caption_path)
    image_captions = group_captions_by_image(caption_df)

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

    # Creating Word Mappings
    # Define special tokens
    special_tokens = ["<pad>", "<start>", "<end>", "<unk>"]
    # Create mappings from words to indices and vice versa
    word2idx = {token: idx for idx, token in enumerate(special_tokens)}
    idx2word = {idx: token for idx, token in enumerate(special_tokens)}

    # Define the maximum vocabulary size
    vocab_size = 10000
    # Get the most common words up to the vocab_size limit
    most_common = word_counts.most_common(vocab_size - len(special_tokens))

    # Add the most common words to the vocabulary
    for idx, (word, _) in enumerate(most_common, start=len(special_tokens)):
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word, image_captions


def convert_captions_to_sequences(image_captions, word2idx):
    # Converting Captions to Sequences
    # Create a dictionary mapping image names to sequences of word indices
    captions_seqs = {}
    max_length = 0  # Keep track of the maximum caption length

    for img_name, captions in image_captions.items():
        seqs = []
        for caption in captions:
            # Tokenize the caption and add <start> and <end> tokens
            tokens = ["<start>"] + tokenize(caption) + ["<end>"]
            # Convert tokens to indices, use <unk> index if word not in vocabulary
            seq = [word2idx.get(token, word2idx["<unk>"]) for token in tokens]
            seqs.append(seq)
            # Update the maximum caption length
            max_length = max(max_length, len(seq))
        captions_seqs[img_name] = seqs

    return captions_seqs, max_length


# Custom Dataset Class for the Flickr30k dataset
class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, image_ids, captions_seqs, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            image_ids (list): List of image filenames.
            captions_seqs (dict): Dictionary mapping image filenames to sequences of word indices.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.captions = []

        # Prepare the list of image and caption pairs
        for img_id in image_ids:
            captions = captions_seqs[img_id]
            for caption_seq in captions:
                self.images.append(img_id)
                self.captions.append(caption_seq)

    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image filename and caption sequence at the given index
        img_id = self.images[idx]
        caption_seq = self.captions[idx]
        # Load the image from the directory
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        # Apply the transformations if any
        if self.transform:
            image = self.transform(image)

        # Convert the caption sequence to a tensor
        caption_seq = torch.tensor(caption_seq)
        return image, caption_seq


# Collate Function for DataLoader
def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    We sort the captions by length to use pack_padded_sequence.
    """
    # Sort data by caption length in descending order
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Stack images into a tensor of shape (batch_size, 3, H, W)
    images = torch.stack(images, 0)
    # Get the lengths of each caption
    lengths = [len(cap) for cap in captions]
    # Create a tensor to hold the padded captions
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        # Copy the caption into the targets tensor
        targets[i, :end] = cap[:end]

    return images, targets, lengths


def get_image_transformations():
    # Image Transformations
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # mean
            ),  # std
        ]
    )
    return transform


def prepare_image2captions(
    image_ids: List[str],
    captions_seqs: Dict[str, List[List[int]]],
    idx2word: Dict[int, str],
) -> Dict[str, List[List[str]]]:
    """
    Prepare a dictionary mapping image IDs to lists of reference captions (each as a list of tokens).

    Args:
        image_ids (List[str]): A list of image IDs.
        captions_seqs (Dict[str, List[List[int]]]): A dictionary mapping image IDs to lists of caption sequences (each sequence is a list of word indices).
        idx2word (Dict[int, str]): A dictionary mapping word indices to words.

    Returns:
        Dict[str, List[List[str]]]: A dictionary mapping image IDs to lists of reference captions (each caption is a list of words).
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


def get_splits(image_names):
    # Splitting the data into training, validation, and test sets
    # Split into training and temporary sets (e.g., 70% training)
    train_images, temp_images = train_test_split(
        image_names, test_size=0.3, random_state=42
    )
    # Split the temporary set into validation and test sets (e.g., 15% each)
    val_images, test_images = train_test_split(
        temp_images, test_size=0.5, random_state=42
    )
    return train_images, val_images, test_images


def get_loaders(img_dir, caption_path, transform):
    # Build vocabulary and get image captions
    word2idx, idx2word, image_captions = build_vocabulary(caption_path)

    # Convert captions to sequences
    captions_seqs, max_length = convert_captions_to_sequences(image_captions, word2idx)
    train_images, val_images, test_images = get_splits(list(image_captions.keys()))

    # Create the datasets
    train_dataset = Flickr30kDataset(img_dir, train_images, captions_seqs, transform)
    val_dataset = Flickr30kDataset(img_dir, val_images, captions_seqs, transform)
    test_dataset = Flickr30kDataset(img_dir, test_images, captions_seqs, transform)

    # Create the DataLoaders
    batch_size = 32  # Define the batch size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


# Function to generate captions for an image
def generate_caption_ids(decoder, features, word2idx, max_length=100):
    """
    Generate a caption for an image given the extracted features.
    """
    sampled_ids = []
    inputs = features.unsqueeze(1)  # Start with the image features as input
    states = None  # Initial hidden states

    for _ in range(max_length):
        # Pass through the LSTM
        hiddens, states = decoder.lstm(inputs, states)
        # Pass through the linear layer to get scores over the vocabulary
        outputs = decoder.linear(hiddens.squeeze(1))
        # Get the most probable word index
        predicted = outputs.argmax(1)
        sampled_ids.append(predicted.item())
        # If the <end> token is generated, stop
        if predicted.item() == word2idx["<end>"]:
            break
        # Prepare input for the next time step
        inputs = decoder.embed(predicted)
        inputs = inputs.unsqueeze(1)

    return sampled_ids


# Function to generate a caption for a given image
def generate_caption(encoder, decoder, image, word2idx, idx2word, max_length=20):
    """
    Generate a caption for a given image using the encoder and decoder models.

    Args:
        encoder: The encoder model.
        decoder: The decoder model.
        image (torch.Tensor): The input image tensor.
        word2idx (Dict[str, int]): A dictionary mapping words to indices.
        idx2word (Dict[int, str]): A dictionary mapping indices to words.
        max_length (int, optional): The maximum length of the generated caption. Defaults to 20.

    Returns:
        str: The generated caption as a string.
    """
    # Set the models to evaluation mode
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Move the image tensor to the appropriate device and add a batch dimension
        image = image.to(device).unsqueeze(0)
        # Extract features from the image using the encoder
        features = encoder(image)
        # Generate word IDs for the caption using the decoder
        sampled_ids = generate_caption_ids(decoder, features, word2idx, max_length)
        # Convert word IDs to words
        sampled_caption = [idx2word.get(word_id, "<unk>") for word_id in sampled_ids]

        # Remove special tokens from the caption
        sampled_caption = [
            word
            for word in sampled_caption
            if word not in ["<start>", "<end>", "<pad>"]
        ]
    # Join the words to form the final caption string
    return " ".join(sampled_caption)
