# data/dataset.py

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

class FlickrDataset(Dataset):
    """
    Custom Dataset for loading Flickr images and captions.
    This dataset handles the basic functionality of loading images and their corresponding captions.
    """

    def __init__(self, image_dir, image_ids, captions_seqs, transform=None, mode='train'):
        """
        Args:
            image_dir (str): Directory with all the images.
            image_ids (list): List of image filenames.
            captions_seqs (dict): Dictionary mapping image filenames to caption sequences.
            transform (callable, optional): Optional transform to be applied on an image.
            mode (str): Mode of the dataset, 'train' or 'test'.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.image_ids = image_ids
        self.captions_seqs = captions_seqs

        if self.mode == 'train':
            self.images = []
            self.captions = []
            for img_id in image_ids:
                captions = captions_seqs[img_id]
                for caption_seq in captions:
                    self.images.append(img_id)
                    self.captions.append(caption_seq)
        elif self.mode == 'test':
            # For testing, we directly work with image_ids and choose one caption per image
            # (You can also handle multiple captions per image if you want)
            pass
        else:
            raise ValueError("Mode should be either 'train' or 'test'.")

    def __len__(self):
        if self.mode == 'train':
            return len(self.images)
        elif self.mode == 'test':
            return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Retrieves the image and caption at the specified index.
        Args:
            idx (int): Index
        Returns:
            image (Tensor): Transformed image tensor.
            caption_seq (Tensor): Corresponding caption sequence tensor.
            image_id (str): Filename of the image.
        """
        if self.mode == 'train':
            img_id = self.images[idx]
            caption_seq = self.captions[idx]
        elif self.mode == 'test':
            img_id = self.image_ids[idx]
            caption_seqs = self.captions_seqs[img_id]
            caption_seq = random.choice(caption_seqs) if caption_seqs else []
        img_path = os.path.join(self.image_dir, img_id)

        # Open and convert image to RGB
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Convert caption sequence to tensor
        caption_seq = torch.tensor(caption_seq)

        if self.mode == 'train':
            return image, caption_seq
        elif self.mode == 'test':
            return image, caption_seq, img_id

def collate_fn(batch):
    """
    Custom collate function to handle variable-length captions.
    This function pads captions to the length of the longest caption in the batch.
    Args:
        batch (list): List of tuples (image, caption_seq) or (image, caption_seq, image_id)
    Returns:
        If training:
            images (Tensor): Batch of images.
            targets (Tensor): Padded caption sequences.
            lengths (list): Original lengths of each caption before padding.
        If testing:
            images (Tensor): Batch of images.
            targets (Tensor): Padded caption sequences.
            image_ids (list): List of image filenames.
    """
    if len(batch[0]) == 3:
        # Test mode
        images, captions, image_ids = zip(*batch)
    else:
        # Train mode
        images, captions = zip(*batch)
        image_ids = None

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    max_length = max(lengths)
    targets = torch.zeros(len(captions), max_length).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    if image_ids is not None:
        return images, targets, image_ids
    else:
        return images, targets, lengths

def get_transform(train=True):
    """
    Returns the image transformations for training or evaluation.
    Args:
        train (bool): Flag indicating whether transformations are for training or evaluation.
    Returns:
        transform (callable): Composed transformations.
    """
    if train:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225],   # ImageNet stds
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225],   # ImageNet stds
                ),
            ]
        )
    return transform