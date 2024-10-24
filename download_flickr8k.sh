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