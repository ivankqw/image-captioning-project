#!/bin/bash

# Download Flickr8k images
wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

# Check if the data directory exists
if [ ! -d "data" ]; then
    mkdir "data"
fi

unzip Flickr8k_Dataset.zip -d data/

rm Flickr8k_Dataset.zip

echo "Flickr8k dataset downloaded."
