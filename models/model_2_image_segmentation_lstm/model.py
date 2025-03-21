import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, embed_size=256):
        super(Encoder, self).__init__()
        # Load a pre-trained Mask R-CNN (ResNet-50-FPN) model
        self.maskrcnn = models.detection.maskrcnn_resnet50_fpn(weights="COCO_V1")
        self.maskrcnn.eval()

        # Freeze all the parameters of the Mask R-CNN
        for param in self.maskrcnn.parameters():
            param.requires_grad = False

        # Adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Number of FPN levels, typically 5 ('0', '1', '2', '3', 'pool')
        num_fpn_levels = 5
        in_features = 256 * num_fpn_levels  # 256 channels per FPN level

        # Fully connected layer to transform concatenated features to embed_size
        self.fc = nn.Linear(in_features, embed_size)
        self.init_weights()

    def init_weights(self):
        # Initialize the weights of the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        """
        Forward pass through the encoder.
        Args:
            images (Tensor): Input images, shape (B, C, H, W)
        Returns:
            embeddings (Tensor): Image embeddings, shape (B, embed_size)
        """
        # Ensure the backbone is in evaluation mode
        self.maskrcnn.eval()
        with torch.no_grad():
            # Extract multi-scale features from the backbone
            features = self.maskrcnn.backbone(images)
            # features is an OrderedDict of: {'0': ..., '1': ..., '2': ..., '3': ..., 'pool': ...}
            # Each value is (B, 256, H', W')

        # Pool and concatenate features from all FPN levels
        pooled_features = []
        for key in sorted(features.keys()):
            f = features[key]  # shape: (B, 256, H', W')
            pooled = self.avgpool(f)  # shape: (B, 256, 1, 1)
            pooled = pooled.view(pooled.size(0), -1)  # shape: (B, 256)
            pooled_features.append(pooled)
        
        # Concatenate pooled features from all levels
        concat_features = torch.cat(pooled_features, dim=1)  # shape: (B, 256 * num_fpn_levels) = (B, 1280)

        # Transform features to the embedding size
        embeddings = self.fc(concat_features)  # shape: (B, embed_size)
        return embeddings

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        # Initialize weights for input, forget, cell, and output gates
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate parameters
        self.W_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # Forget gate parameters
        self.W_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # Cell gate parameters
        self.W_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        # Output gate parameters
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_weights()

    def init_weights(self):
        # Initialize all weights and biases
        for param in self.parameters():
            if param.data.ndimension() >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.zeros_(param.data)

    def forward(self, x, h_prev, c_prev):
        # Compute gates
        i_t = torch.sigmoid(x @ self.W_i + h_prev @ self.U_i + self.b_i)
        f_t = torch.sigmoid(x @ self.W_f + h_prev @ self.U_f + self.b_f)
        g_t = torch.tanh(x @ self.W_c + h_prev @ self.U_c + self.b_c)
        o_t = torch.sigmoid(x @ self.W_o + h_prev @ self.U_o + self.b_o)

        # Update cell state
        c_t = f_t * c_prev + i_t * g_t
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class Decoder(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=8000, dropout=0.5):
        super(Decoder, self).__init__()
        # Embedding layer to convert word indices to embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Custom LSTM cell
        self.lstm_cell = LSTM(embed_size, hidden_size)
        # Fully connected layer to project hidden state to vocabulary space
        self.fc = nn.Linear(hidden_size, vocab_size)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        # Initialize weights for embedding and fully connected layers
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, features, captions):
        """
        Forward pass through the decoder.
        Args:
            features: Image features from the encoder, shape (batch_size, embed_size)
            captions: Caption sequences, shape (batch_size, max_seq_length)
        Returns:
            outputs: Predicted word distributions, shape (batch_size, seq_len, vocab_size)
        """
        # Embed the captions (exclude the last word for teacher forcing)
        embeddings = self.embedding(captions[:, :-1])  # Shape: (batch_size, seq_len - 1, embed_size)
        # Concatenate image features as the first input
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        embeddings = self.dropout(embeddings)

        batch_size, seq_len, _ = embeddings.size()
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(features.device)

        # Initialize hidden and cell states to zeros
        h_t = torch.zeros(batch_size, self.hidden_size).to(features.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(features.device)

        # Unroll the LSTM for seq_len time steps
        for t in range(seq_len):
            x_t = embeddings[:, t, :]  # Input at time step t
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)  # Update hidden and cell states
            output = self.fc(h_t)  # Compute output word distribution
            outputs[:, t, :] = output  # Store output

        return outputs

    def sample(self, features, max_len=50, end_token_idx=None):
        """
        Generate captions for given image features using greedy search.
        Args:
            features: Image features from the encoder, shape (batch_size, embed_size)
            max_len: Maximum length of the generated caption
            end_token_idx: Index of the <end> token
        Returns:
            sampled_ids: List of predicted word indices
        """
        sampled_ids = []
        inputs = features  # Initial input is the image features
        device = features.device  # Get the device of the input features

        # Initialize hidden and cell states on the same device as features
        h_t = torch.zeros(features.size(0), self.hidden_size).to(device)
        c_t = torch.zeros(features.size(0), self.hidden_size).to(device)

        for _ in range(max_len):
            h_t, c_t = self.lstm_cell(inputs, h_t, c_t)
            outputs = self.fc(h_t)  # Compute word distribution
            predicted = outputs.argmax(1)  # Get the index with the highest probability
            sampled_ids.append(predicted.item())

            if predicted.item() == end_token_idx:
                break  # Stop if <end> token is generated

            # Prepare input for next time step
            inputs = self.embedding(predicted)
            inputs = self.dropout(inputs)

        return sampled_ids