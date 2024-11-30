from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import random
import torch
from PIL import Image
from torchvision.transforms import transforms
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Define model paths
model_paths = {
    "baseline": {
        "encoder": "models/model_1_baseline_cnn_lstm/encoder.pth",
        "decoder": "models/model_1_baseline_cnn_lstm/decoder.pth",
    },
    "butd_attention": {
        "encoder": "models/model_1.5_butd_attention/encoder.pth",
        "decoder": "models/model_1.5_butd_attention/decoder.pth",
    },
    "segmentation": {
        "encoder": "models/model_2_image_segmentation_lstm/encoder.pth",
        "decoder": "models/model_2_image_segmentation_lstm/decoder.pth",
    },
    "attention": {
        "encoder": "models/model_3_attention_image_segmentation/encoder.pth",
        "decoder": "models/model_3_attention_image_segmentation/decoder.pth",
    },
    "vision_transformer": {
        "encoder": "models/model_4_vision_transformer/encoder.pth",
        "decoder": "models/model_4_vision_transformer/decoder.pth",
    },
}

# Global Variables
image_captions = None
word2idx, idx2word = None, None
captions_seqs = None

# Paths for Flickr8k Dataset
flickr_data_path = "flickr_data/Flickr8k_Dataset/Images"
captions_file = "flickr_data/Flickr8k_Dataset/captions.txt"

# Load and preprocess captions
from data.preprocessing import (
    build_vocabulary,
    convert_captions_to_sequences,
    prepare_image2captions,
)


def preprocess_captions(captions_file):
    caption_df = pd.read_csv(captions_file)
    if not {"image", "caption"}.issubset(caption_df.columns):
        raise ValueError(
            "The captions file must contain 'image' and 'caption' columns."
        )
    return caption_df


caption_df = preprocess_captions(captions_file)
word2idx, idx2word, image_captions = build_vocabulary(caption_df)
captions_seqs, max_length = convert_captions_to_sequences(image_captions, word2idx)

# Image preprocessing
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Define a function to initialize models based on model_name
def initialize_model(model_name):
    vocab_size = len(word2idx)
    embed_size = 256
    hidden_size = 512

    if model_name == "baseline":
        from models.model_1_baseline_cnn_lstm.model import EncoderCNN, DecoderRNN

        encoder = EncoderCNN(embed_size)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    elif model_name == "segmentation":
        from models.model_2_image_segmentation_lstm.model import EncoderCNN, DecoderRNN

        encoder = EncoderCNN(embed_size)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    elif model_name == "attention":
        from models.model_3_attention_image_segmentation_lstm.model import (
            EncoderCNN,
            AttentionDecoderRNN,
        )

        encoder = EncoderCNN(embed_size)
        decoder = AttentionDecoderRNN(embed_size, hidden_size, vocab_size)
    elif model_name == "vision_transformer":
        from models.model_4_vision_transformer.model import (
            VisionTransformerEncoder,
            DecoderRNN,
        )

        encoder = VisionTransformerEncoder(embed_size)
        decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
    else:
        raise ValueError("Invalid model name")
    return encoder, decoder


# Load model function
def load_model(model_name):
    if model_name not in model_paths:
        raise HTTPException(status_code=400, detail="Invalid model name")

    try:
        encoder, decoder = initialize_model(model_name)
    except Exception as e:
        print(f"Error initializing model instances for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing model: {e}")

    try:
        encoder_state_dict = torch.load(
            model_paths[model_name]["encoder"], map_location=torch.device("cpu")
        )
        decoder_state_dict = torch.load(
            model_paths[model_name]["decoder"], map_location=torch.device("cpu")
        )

        encoder.load_state_dict(encoder_state_dict)
        decoder.load_state_dict(decoder_state_dict)

        encoder.eval()
        decoder.eval()
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    return encoder, decoder


# Predict caption function
def predict_caption(encoder, decoder, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample(features)

    # Convert word IDs to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = idx2word.get(word_id.item(), "<unk>")
        sampled_caption.append(word)
        if word == "<end>":
            break
    caption = " ".join(sampled_caption[1:-1])  # Exclude <start> and <end> tokens
    return caption


# Showcase 9 random images
@app.get("/showcase")
async def showcase(model_name: str):
    if model_name not in model_paths:
        raise HTTPException(status_code=400, detail="Invalid model name")

    encoder, decoder = load_model(model_name)
    random_images = random.sample(list(image_captions.keys()), 9)
    results = []

    for image_name in random_images:
        image_path = os.path.join(flickr_data_path, image_name)
        predicted_caption = predict_caption(encoder, decoder, image_path)
        actual_captions = [" ".join(caption) for caption in image_captions[image_name]]
        results.append(
            {
                "image": image_name,
                "predicted_caption": predicted_caption,
                "actual_captions": actual_captions,
            }
        )
    return results


# Upload an image for captioning
@app.post("/upload")
async def upload_image(model_name: str, file: UploadFile = File(...)):
    if model_name not in model_paths:
        raise HTTPException(status_code=400, detail="Invalid model name")

    # Save uploaded image temporarily
    temp_image_path = f"temp_{file.filename}"
    with open(temp_image_path, "wb") as buffer:
        buffer.write(await file.read())

    encoder, decoder = load_model(model_name)
    try:
        predicted_caption = predict_caption(encoder, decoder, temp_image_path)
    finally:
        os.remove(temp_image_path)

    return {"predicted_caption": predicted_caption}
