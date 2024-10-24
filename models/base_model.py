import torch
import torch.nn as nn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        """
        Simple CNN encoder for image feature extraction.
        """
        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Second convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Third convolutional layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Fully connected layer
        self.embed = nn.Linear(256, embed_size)
        self.init_weights()

    def init_weights(self):
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        features = self.features(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Linear layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        # Initialize weights
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, features, captions):
        # Embed the captions
        embeddings = self.embed(captions[:, :-1])
        # Concatenate features and embeddings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # Apply dropout
        embeddings = self.dropout(embeddings)
        # LSTM forward
        hiddens, _ = self.lstm(embeddings)
        # Output layer
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, states=None, max_len=20, end_token_idx=None):
        """
        Generate a caption for an image given the extracted features.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
            if end_token_idx is not None and predicted.item() == end_token_idx:
                break
        return sampled_ids
