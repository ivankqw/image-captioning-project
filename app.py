import streamlit as st
import requests
import pandas as pd
from PIL import Image
import io
from data.preprocessing import (
    tokenize, build_vocabulary, convert_captions_to_sequences,
    get_splits, prepare_image2captions
)

# API Base URL
API_URL = "http://127.0.0.1:8000"

# Load the Flickr8k captions CSV file for preprocessing
caption_df = pd.read_csv("flickr_data/Flickr8k_Dataset/captions.txt") 

# Build vocabulary and prepare caption mappings
word2idx, idx2word, image_captions = build_vocabulary(caption_df)
captions_seqs, max_length = convert_captions_to_sequences(image_captions, word2idx)
image_names = list(image_captions.keys())
train_images, val_images, test_images = get_splits(image_names)
image2captions = prepare_image2captions(image_names, captions_seqs, idx2word)

# Streamlit App
st.title("CS7643 Image Captioning Project - CaptionWhisperer")

# Dropdown to select a model
model_name = st.selectbox("Select a Model", ["baseline", "segmentation", "attention", "vision_transformer"])

# Showcase 9 Random Images
if st.button("Show Random Images"):
    st.header("Random Image Showcase")
    response = requests.get(f"{API_URL}/showcase", params={"model_name": model_name})
    if response.status_code == 200:
        results = response.json()
        cols = st.columns(3)
        for i, result in enumerate(results):
            with cols[i % 3]:
                image = Image.open(f"flickr_data/images/{result['image']}")
                st.image(image, caption=f"Predicted: {result['predicted_caption']}")
                st.write(f"Actual: {', '.join(result['actual_captions'])}")
    else:
        st.error("Failed to fetch showcase images.")

# Upload an Image for Prediction
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.header("Upload an Image for Caption Prediction")
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image")
    response = requests.post(
        f"{API_URL}/upload",
        files={"file": uploaded_image.getvalue()},
        data={"model_name": model_name},
    )
    if response.status_code == 200:
        prediction = response.json()["predicted_caption"]
        st.write(f"Predicted Caption: {prediction}")
    else:
        st.error("Failed to generate a caption for the uploaded image.")
