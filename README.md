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
sh download_flickr8k.sh
```

To download the Flickr30k dataset
```bash
sh download_flickr30k.sh
```

## Demo Usage

Feel free to select any images and see what the different model caption them as! 

Run the following code:

```bash
uvicorn app.main:app --reload
```

Enter this into your address bar: http://127.0.0.1:8000
