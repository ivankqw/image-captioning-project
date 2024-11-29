# image-captioning-project
deep learning image captioning project 

## Setup

Virtual environment setup
```bash
conda env create -f conda.yml
conda activate image-captioning-project
```

Or if you already have the conda env, you can update it by running this:
```bash
conda env update --file conda.yml --prune
```

Install the precommit hook 
```bash
pre-commit install
```

To download the Flickr8k dataset
```bash
sh scripts/download_flickr8k.sh
```

To download the Flickr30k dataset
```bash
sh scripts/download_flickr30k.sh
```

## Usage

Training

Change the --model arg accordingly:
1. model_1_baseline_cnn_lstm - python main.py --model model_1_baseline_cnn_lstm --mode train --dataset Flickr8k 
2. model_2_image_segmentation_lstm - python main.py --model model_2_image_segmentation_lstm --mode train --dataset Flickr8k 
3. model_3_attention_image_segmentation_lstm - python main.py --model model_3_attention_image_segmentation_lstm --mode train --dataset Flickr8k 
4. model_4_vision_transformer - python main.py --model model_4_vision_transformer --mode train --dataset Flickr8k 

To train the model on Flickr8k dataset:
```bash
python main.py --model model_1_baseline_cnn_lstm --mode train --dataset Flickr8k 
```

To train the model on Flickr30k dataset:
```bash
python main.py --model model_1_baseline_cnn_lstm --mode train --dataset Flickr30k
```

To train the model on Flickr8k dataset:
```bash
python main.py --model model_1_baseline_cnn_lstm --mode test --dataset Flickr8k
```

To train the model on Flickr30k dataset:
```bash
python main.py --model model_1_baseline_cnn_lstm --mode test --dataset Flickr30k
```