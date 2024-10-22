from PIL import Image
from textwrap import wrap
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from cfg import flickr_image_path, flickr_text_path
import torch


# Function to read and preprocess the image using PyTorch (PIL)
def readImage(path, img_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts image to PyTorch Tensor and normalizes between 0 and 1
        ]
    )
    img = Image.open(path).convert("RGB")
    img = transform(img)
    return img.permute(
        1, 2, 0
    ).numpy()  # Permute to (H, W, C) for matplotlib compatibility


# Function to display images along with captions
def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    n = 0
    for i in range(15):  # Display 15 images
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)

        # Load and display the image
        image_path = f"{flickr_image_path}/{temp_df.image[i]}"
        image = readImage(image_path)
        plt.imshow(image)

        # Display the caption
        caption = "\n".join(
            wrap(temp_df.caption[i], 20)
        )  # Wrap text for better display
        plt.title(caption)
        plt.axis("off")  # Turn off axis

    plt.show()


def plot_results(
    num_epochs, train_losses, val_losses, val_bleu_scores, val_meteor_scores
):
    # Plotting the Loss Curves
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()

    # Plotting BLEU and METEOR Scores
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, val_bleu_scores, label="Validation BLEU Score")
    plt.plot(epochs, val_meteor_scores, label="Validation METEOR Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation BLEU and METEOR Scores Over Epochs")
    plt.legend()
    plt.show()
