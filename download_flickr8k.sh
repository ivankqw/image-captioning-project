#!/bin/bash

# Download Flickr8k dataset
wget "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr8k.zip"

# Check if the data directory exists
if [ ! -d "flickr_data" ]; then
    mkdir "flickr_data"
fi

# Unzip the dataset into the flickr_data/Flickr8k_Dataset directory
unzip -q flickr8k.zip -d flickr_data/Flickr8k_Dataset

# Remove the zip file after extraction
rm flickr8k.zip
echo "Downloaded Flickr8k dataset successfully."

# Check if GloVe embeddings exist, download if not
GLOVE_DIR="glove"
GLOVE_FILE="glove.6B.50d.txt"

if [ ! -f "$GLOVE_DIR/$GLOVE_FILE" ]; then
    echo "GloVe embeddings not found. Downloading..."
    
    # Create the glove directory if it doesn't exist
    if [ ! -d "$GLOVE_DIR" ]; then
        mkdir "$GLOVE_DIR"
    fi

    # Download GloVe embeddings
    wget "http://nlp.stanford.edu/data/glove.6B.zip" -P "$GLOVE_DIR"
    
    # Unzip and keep only the 50d version
    unzip -q "$GLOVE_DIR/glove.6B.zip" -d "$GLOVE_DIR"
    rm "$GLOVE_DIR/glove.6B.zip"
    echo "Downloaded and extracted GloVe embeddings successfully."
else
    echo "GloVe embeddings already exist. Skipping download."
fi