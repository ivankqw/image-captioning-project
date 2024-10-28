import os
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse

from models.base_model import EncoderCNN, DecoderRNN
from utils import build_vocab, captions_to_sequences, prepare_image2captions, load_glove_embeddings

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Test image captioning model.')
    parser.add_argument('--dataset', type=str, required=True, choices=['Flickr8k', 'Flickr30k'],
                        help='Specify which dataset to use: Flickr8k or Flickr30k')
    args = parser.parse_args()

    # Paths
    dataset_dir = f'./flickr_data/{args.dataset}_Dataset/Images'
    captions_file = f'./flickr_data/{args.dataset}_Dataset/captions.txt'
    image_dir = dataset_dir
    glove_path = "./glove/glove.6B.200d.txt"  # Make sure this path is correct

    # Prepare captions and build vocabulary
    caption_df = pd.read_csv(captions_file).dropna().drop_duplicates()
    image_captions = caption_df.groupby("image")["caption"].apply(list).to_dict()

    # Build vocabulary from captions
    vocab_size = 5000
    word2idx, idx2word = build_vocab(image_captions, vocab_size)  # Assume build_vocab is available

    # Load GloVe embeddings
    embed_size = 200
    embedding_matrix = load_glove_embeddings(glove_path, word2idx, vocab_size, embed_size)

    # Convert captions to sequences
    captions_seqs = captions_to_sequences(image_captions, word2idx)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))
    ])

    # Split data
    image_names = list(image_captions.keys())
    _, temp_images = train_test_split(image_names, test_size=0.3, random_state=42)
    _, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    # Prepare image-to-captions mapping
    test_image2captions = prepare_image2captions(test_images, captions_seqs, idx2word)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Models
    encoder = EncoderCNN(embed_size=embed_size).to(device)
    decoder = DecoderRNN(embed_size=embed_size, hidden_size=512, vocab_size=len(word2idx), embedding_matrix=embedding_matrix, num_layers=2).to(device)

    # Load trained models from the 'models/' directory
    encoder.load_state_dict(torch.load('models/base_model_encoder.pth', map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load('models/base_model_decoder.pth', map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()

    end_token_idx = word2idx['<end>']

    # Generate captions
    for img_id in test_images[:10]:
        img_path = os.path.join(image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(image_tensor)
            sampled_ids = decoder.sample(features, end_token_idx=end_token_idx)
            sampled_caption = [idx2word.get(word_id, '<unk>') for word_id in sampled_ids]
            sampled_caption = [word.lower() for word in sampled_caption if word not in ['<start>', '<end>', '<pad>']]

        print(f'Image: {img_id}')
        print('Generated Caption:', ' '.join(sampled_caption))
        print('Ground Truth Captions:')
        for ref in test_image2captions[img_id]:
            print(' '.join(ref))
        print('-' * 50)

if __name__ == '__main__':
    main()