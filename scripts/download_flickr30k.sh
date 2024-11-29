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
if [ ! -d "../flickr_data" ]; then
    mkdir "../flickr_data"
fi

# Unzip the Flickr30k dataset into a flickr30k directory
unzip -q flickr30k.zip -d ../flickr_data/Flickr30k_Dataset

# Remove the zip file
rm flickr30k.zip
echo "Downloaded Flickr30k dataset successfully."

