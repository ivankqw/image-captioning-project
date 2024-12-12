import os
import random
import importlib.util
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

from data.dataset import *
from data.preprocessing import *

app = FastAPI()

# Define the base directory
BASE_DIR = Path(__file__).resolve().parent.parent  # project_root

# Paths for Flickr8k Dataset
flickr_data_path = BASE_DIR / "flickr_data" / "Flickr8k_Dataset" / "Images"
captions_file = BASE_DIR / "flickr_data" / "Flickr8k_Dataset" / "captions.txt"

# Ensure the uploads directory exists
UPLOAD_DIR = BASE_DIR / "app" / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static directories
app.mount(
    "/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static"
)
app.mount(
    "/flickr_images", StaticFiles(directory=str(flickr_data_path)), name="flickr_images"
)

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# Load captions
caption_df = pd.read_csv(captions_file).dropna().drop_duplicates()

# Build vocabulary once (adjust vocab_size if needed)
vocab_size = 8000
word2idx, idx2word, image_captions = build_vocabulary(caption_df, vocab_size=vocab_size)

# Convert captions to sequences
captions_seqs, max_length = convert_captions_to_sequences(image_captions, word2idx)

# Transformation (ensure get_transform is defined)
test_transform = get_transform(train=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths dictionary
model_paths = {
    "baseline": {
        "encoder": str(
            BASE_DIR / "models" / "model_1_baseline_cnn_lstm" / "encoder.pth"
        ),
        "decoder": str(
            BASE_DIR / "models" / "model_1_baseline_cnn_lstm" / "decoder.pth"
        ),
        "model": str(BASE_DIR / "models" / "model_1_baseline_cnn_lstm" / "model.py"),
    },
    "segmentation": {
        "encoder": str(
            BASE_DIR / "models" / "model_2_image_segmentation_lstm" / "encoder.pth"
        ),
        "decoder": str(
            BASE_DIR / "models" / "model_2_image_segmentation_lstm" / "decoder.pth"
        ),
        "model": str(
            BASE_DIR / "models" / "model_2_image_segmentation_lstm" / "model.py"
        ),
    },
    "attention": {
        "encoder": str(
            BASE_DIR
            / "models"
            / "model_3_image_segmentation_attention_decoder"
            / "encoder.pth"
        ),
        "decoder": str(
            BASE_DIR
            / "models"
            / "model_3_image_segmentation_attention_decoder"
            / "decoder.pth"
        ),
        "model": str(
            BASE_DIR
            / "models"
            / "model_3_image_segmentation_attention_decoder"
            / "model.py"
        ),
    },
    # "vision_transformer": {
    #     "encoder": str(
    #         BASE_DIR / "models" / "model_4_vision_transformer" / "encoder.pth"
    #     ),
    #     "decoder": str(
    #         BASE_DIR / "models" / "model_4_vision_transformer" / "decoder.pth"
    #     ),
    #     "model": str(BASE_DIR / "models" / "model_4_vision_transformer" / "model.py"),
    # },
}

current_model_type = "attention"

# Global references to encoder and decoder
encoder = None
decoder = None


def load_model(model_type: str):
    global encoder, decoder

    model_file_path = model_paths[model_type]["model"]
    encoder_path = model_paths[model_type]["encoder"]
    decoder_path = model_paths[model_type]["decoder"]

    # Dynamically import the model module
    spec = importlib.util.spec_from_file_location("model_mod", model_file_path)
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)

    # Classes should be defined in the imported module (e.g., EncoderCNN, DecoderRNN)
    EncoderClass = getattr(model_mod, "Encoder", None)
    DecoderClass = getattr(model_mod, "Decoder", None)

    if EncoderClass is None or DecoderClass is None:
        raise ValueError(f"Encoder or Decoder not found in {model_file_path}")

    # Initialize with appropriate sizes - adjust if your model classes require different params
    embed_size = 256
    hidden_size = 512

    encoder = EncoderClass(embed_size=embed_size).to(device)
    decoder = DecoderClass(
        embed_size=embed_size, hidden_size=hidden_size, vocab_size=len(word2idx)
    ).to(device)

    # Load weights
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()


# Initially load the default model
load_model(current_model_type)


def generate_caption_from_image_tensor(image_tensor):
    with torch.no_grad():
        features = encoder(image_tensor.unsqueeze(0).to(device))
        end_token_idx = word2idx.get("<end>", None)
        sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
    sampled_caption = [idx2word.get(word_id, "<unk>") for word_id in sampled_ids]
    if "<end>" in sampled_caption:
        end_index = sampled_caption.index("<end>")
        sampled_caption = sampled_caption[:end_index]
    generated_caption = " ".join(sampled_caption)
    return generated_caption


def generate_caption_for_image(image_path: str, is_vit_gpt2=False):
    if is_vit_gpt2:
        # hacky way for now to use the Vision Transformer + GPT-2 model
        # this will be refactored in the future (or not...)
        from transformers import GPT2TokenizerFast, VisionEncoderDecoderModel

        model = VisionEncoderDecoderModel.from_pretrained(
            "ivankqw/image-captioning-vit-gpt2"
        )
        tokenizer = GPT2TokenizerFast.from_pretrained(
            "ivankqw/image-captioning-vit-gpt2"
        )
        from torchvision import transforms

        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=0.5, std=0.5  # we know the mean and std for vit
                ),
            ]
        )
        pix = test_transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
        print(f"running inference on image: {image_path} with vit_gpt2 model")
        return tokenizer.decode(
            model.generate(
                pix,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=1,
                do_sample=False,
                min_length=10,
                max_length=25,
            ).sequences[0],
            skip_special_tokens=True,
        )
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image)
    caption = generate_caption_from_image_tensor(image_tensor)
    return caption


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request, model: str = "baseline"):
    if model in model_paths:
        load_model(model)
    elif model == "vision_transformer":
        image_names = list(image_captions.keys())
        sampled_test_images = random.sample(image_names, 9)
        images_with_captions = []
        for img_name in sampled_test_images:
            img_path = flickr_data_path / img_name
            caption = generate_caption_for_image(str(img_path), is_vit_gpt2=True)
            images_with_captions.append((img_name, caption))
        return templates.TemplateResponse(
            "gallery.html",
            {
                "request": request,
                "images_with_captions": images_with_captions,
                "model": model,
            },
        )
    else:
        model = "baseline"  # fallback if model not found

    # Select 9 random images from dataset
    image_names = list(image_captions.keys())
    sampled_test_images = random.sample(image_names, 9)

    images_with_captions = []
    for img_name in sampled_test_images:
        img_path = flickr_data_path / img_name
        caption = generate_caption_for_image(str(img_path))
        images_with_captions.append((img_name, caption))

    return templates.TemplateResponse(
        "gallery.html",
        {
            "request": request,
            "images_with_captions": images_with_captions,
            "model": model,
        },
    )


@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    # List uploaded images
    uploaded_images = [
        f.name
        for f in UPLOAD_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
    ]
    return templates.TemplateResponse(
        "upload.html", {"request": request, "uploaded_images": uploaded_images}
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    # Ensure the filename is unique to prevent overwriting
    filename = file.filename
    upload_path = UPLOAD_DIR / filename
    counter = 1
    while upload_path.exists():
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{counter}{ext}"
        upload_path = UPLOAD_DIR / filename
        counter += 1

    # Save the uploaded file
    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # Add the uploaded image to the dropdown list
    uploaded_images = [
        f.name
        for f in UPLOAD_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
    ]

    # Pass the list of uploaded images to the template
    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "uploaded_images": uploaded_images,
            "uploaded_image": None,  # No image displayed
            "generated_caption": None,  # No caption generated
        },
    )


@app.post("/generate_caption", response_class=HTMLResponse)
async def generate_caption(
    request: Request, image_name: str = Form(...), model: str = Form(...)
):
    # Path to the uploaded image
    image_path = UPLOAD_DIR / image_name
    if not image_path.exists():
        # List uploaded images
        uploaded_images = [
            f.name
            for f in UPLOAD_DIR.iterdir()
            if f.is_file()
            and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
        ]
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "error": "Selected image does not exist.",
                "uploaded_images": uploaded_images,
            },
        )
    if model in model_paths:
        load_model(model)
    elif model == "vision_transformer":
        caption = generate_caption_for_image(image_path, is_vit_gpt2=True)
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "uploaded_image": image_url,
                "generated_caption": caption,
                "model": model,
                "uploaded_images": uploaded_images,
            },
        )
    else:
        model = "baseline"

    # Generate caption
    caption = generate_caption_for_image(str(image_path))

    # Image URL
    image_url = f"/static/uploads/{image_name}"

    # Pass the list of uploaded images
    uploaded_images = [
        f.name
        for f in UPLOAD_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
    ]

    return templates.TemplateResponse(
        "upload.html",
        {
            "request": request,
            "uploaded_image": image_url,
            "generated_caption": caption,
            "model": model,
            "uploaded_images": uploaded_images,
        },
    )
