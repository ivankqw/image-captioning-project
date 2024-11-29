# test_model_loading.py
import torch

# Replace with the appropriate model import
from models.model_1_baseline_cnn_lstm.model import EncoderCNN, DecoderRNN

# Initialize models
encoder = EncoderCNN(embed_size=256)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=5000)

# Load state dicts
encoder_state_dict = torch.load(
    'models/model_1_baseline_cnn_lstm/encoder.pth', map_location=torch.device('cpu')
)
decoder_state_dict = torch.load(
    'models/model_1_baseline_cnn_lstm/decoder.pth', map_location=torch.device('cpu')
)

# Load state dicts into models
encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

print("Models loaded successfully.")