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