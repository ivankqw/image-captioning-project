import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=200):
        super(EncoderCNN, self).__init__()
        # Initialize ResNet-50 without pre-trained weights
        resnet = models.resnet50(pretrained=False)
        # Remove the last fully connected layer
        modules = list(resnet.children())[:-1]  # Remove the last fc layer
        self.resnet = nn.Sequential(*modules)
        # Add a new fully connected layer for embedding
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.init_weights()

    def init_weights(self):
        # Initialize the embedding layer weights
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.zeros_(self.embed.bias)

    def forward(self, images):
        # Extract features from the images
        features = self.resnet(images)  # Output shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # Shape: (batch_size, 2048)
        features = self.embed(features)  # Shape: (batch_size, embed_size)
        return features

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """ 
        Init function for LSTM class
        Args:
            input_size (int): the number of features in the inputs.
            hidden_size (int): the size of the hidden layer
        Returns: 
            None
        """
        super(LSTM, self).__init__()

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
        self.W_g = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))
        
        # Output gate parameters
        self.W_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.init_hidden()

    def init_hidden(self):
        # Initialize weights
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, h_t, c_t):
        """
        Forward pass for a single time step
        Args:
            x (torch.Tensor): Input at current time step (batch, input_size)
            h_t (torch.Tensor): Hidden state from previous time step (batch, hidden_size)
            c_t (torch.Tensor): Cell state from previous time step (batch, hidden_size)
        Returns:
            h_t (torch.Tensor): Updated hidden state (batch, hidden_size)
            c_t (torch.Tensor): Updated cell state (batch, hidden_size)
        """
        # Input gate
        i_t = torch.sigmoid(x @ self.W_i + h_t @ self.U_i + self.b_i)
        
        # Forget gate
        f_t = torch.sigmoid(x @ self.W_f + h_t @ self.U_f + self.b_f)
        
        # Cell gate
        g_t = torch.tanh(x @ self.W_g + h_t @ self.U_g + self.b_g)
        
        # Output gate
        o_t = torch.sigmoid(x @ self.W_o + h_t @ self.U_o + self.b_o)
        
        # Update cell state
        c_t = f_t * c_t + i_t * g_t
        
        # Update hidden state
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size=200,
        hidden_size=512,
        vocab_size=5000,
        num_layers=1,
        dropout=0.5,
    ):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size)  # Using the custom LSTM
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(
            self.embedding.weight, -0.1, 0.1
        )  # Initialize embedding weights
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, features, captions):
        """
        Forward pass for the decoder.
        Args:
            features (torch.Tensor): Image features from the encoder (batch, embed_size)
            captions (torch.Tensor): Ground truth captions (batch, max_length)
        Returns:
            outputs (torch.Tensor): Predicted word scores (batch, max_length, vocab_size)
        """
        embeddings = self.embedding(captions[:, :-1])  # (batch, max_length-1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # (batch, max_length, embed_size)
        embeddings = self.dropout(embeddings)

        batch_size, seq_len, _ = embeddings.size()
        outputs = torch.zeros(batch_size, seq_len, self.linear.out_features).to(embeddings.device)

        # Initialize hidden and cell states to zeros on the correct device
        h_t = torch.zeros(batch_size, self.hidden_size).to(embeddings.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(embeddings.device)

        for t in range(seq_len):
            x_t = embeddings[:, t, :]  # (batch, embed_size)
            h_t, c_t = self.lstm(x_t, h_t, c_t)  # (batch, hidden_size), (batch, hidden_size)
            output = self.linear(h_t)  # (batch, vocab_size)
            outputs[:, t, :] = output

        return outputs

    def sample(self, features, max_len=20, end_token_idx=None):
        """
        Generate a caption for an image given the extracted features.
        Uses greedy search to generate the caption.
        Args:
            features (torch.Tensor): Image features from the encoder (1, embed_size)
            max_len (int): Maximum length of the caption
            end_token_idx (int): Index of the <end> token
        Returns:
            sampled_ids (list): List of word indices representing the generated caption
        """
        sampled_ids = []
        inputs = features  # (1, embed_size)
        h_t = torch.zeros(1, self.hidden_size).to(features.device)
        c_t = torch.zeros(1, self.hidden_size).to(features.device)

        for _ in range(max_len):
            h_t, c_t = self.lstm(inputs, h_t, c_t)  # (1, hidden_size), (1, hidden_size)
            outputs = self.linear(h_t)  # (1, vocab_size)
            _, predicted = outputs.max(1)  # (1)
            sampled_ids.append(predicted.item())  # Append the predicted word index

            if predicted.item() == end_token_idx:
                break

            inputs = self.embedding(predicted)  # (1, embed_size)
            inputs = self.dropout(inputs)  # Apply dropout

        return sampled_ids