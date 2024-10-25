#!/bin/bash

# Download Flickr30k images in parts
wget https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00
wget https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01
wget https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02

# Concatenate the parts into a single zip file
cat flickr30k_part00 flickr30k_part01 flickr30k_part02 > flickr30k.zip

# Remove the individual parts
rm flickr30k_part00 flickr30k_part01 flickr30k_part02

# Check if the data directory exists
if [ ! -d "flickr_data" ]; then
    mkdir "flickr_data"
fi

# Unzip the Flickr30k dataset into a flickr30k directory
unzip -q flickr30k.zip -d flickr_data/Flickr30k_Dataset

# Remove the zip file
rm flickr30k.zip
echo "Downloaded Flickr30k dataset successfully."

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