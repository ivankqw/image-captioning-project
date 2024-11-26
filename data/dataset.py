import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FlickrDataset(Dataset):
    def __init__(self, image_dir, image_ids, captions_seqs, transform=None):
        """
        Custom Dataset for loading Flickr images and captions.
        Args:
            image_dir: Directory with all the images.
            image_ids: List of image filenames.
            captions_seqs: Dictionary mapping image filenames to caption sequences.
            transform: Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.captions = []

        # Prepare the dataset by pairing images with their captions
        for img_id in image_ids:
            captions = captions_seqs[img_id]
            for caption_seq in captions:
                self.images.append(img_id)
                self.captions.append(caption_seq)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.images[idx]
        caption_seq = self.captions[idx]
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption_seq = torch.tensor(caption_seq)
        return image, caption_seq

def collate_fn(batch):
    """
    Custom collate function to handle variable length captions.
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_transform(train=True):
    """
    Get image transformations for training or evaluation.
    """
    if train:
        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
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
