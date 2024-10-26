import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class FlickrDataset(Dataset):
    def __init__(self, image_dir, image_ids, captions_seqs, transform=None):
        """
        Custom dataset for Flickr images and captions.
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
        # Load the image
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Convert the caption sequence to a tensor
        caption_seq = torch.tensor(caption_seq)
        return image, caption_seq

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (image, caption).
    """
    # Sort data by caption length in descending order
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Stack images into a tensor
    images = torch.stack(images, 0)
    # Get the lengths of each caption
    lengths = [len(cap) for cap in captions]
    # Create a tensor to hold the padded captions
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths
