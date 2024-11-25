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
To download the Flickr8k dataset
```bash
sh download_flickr8k.sh
```

To download the Flickr30k dataset
```bash
sh download_flickr30k.sh
```

## Usage

Training

Change the --model accordingly:
1. base_model
2. transfer_learning_model
3. transformer_model
4. advance_transformer_model

Train the model on Flickr8k dataset:
```bash
python main.py --model base_model --mode train --dataset Flickr8k 
```

Train the model on Flickr30k dataset:
```bash
python main.py --model base_model --mode train --dataset Flickr30k
```

Test the model on Flickr8k dataset:
```bash
python main.py --model base_model --mode test --dataset Flickr8k
```

Test the model on Flickr30k dataset:
```bash
python main.py --model base_model --mode test --dataset Flickr30k
```