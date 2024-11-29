#!/bin/bash

# https://github.com/shilrley6/Faster-R-CNN-with-model-pretrained-on-Visual-Genome/tree/master
# Download the faster_rcnn_resnet101_vg model
MODEL_PATH="../models/model_1.5_bottom_up_attention"
MODEL_URL="https://drive.google.com/uc?export=download&id=18n_3V1rywgeADZ3oONO0DsuuS9eMW6sN"
MODEL_FILE="faster_rcnn_resnet101_vg.pth"

# Check if the models directory exists
if [ ! -d $MODEL_PATH ]; then
    mkdir $MODEL_PATH
fi

# Download the model
wget -O "$MODEL_PATH/$MODEL_FILE" "$MODEL_URL"
echo "Downloaded $MODEL_FILE successfully."