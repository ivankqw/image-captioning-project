import torch
import random

# Setting up the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure reproducibility
torch.manual_seed(42)
random.seed(42)

flickr_image_path = "data/Flicker8k_Dataset"
flickr_text_path = "data/captions.txt"
