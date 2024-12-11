import torch
import torch.nn as nn
import torchvision.models as models
import math

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

class Decoder(nn.Module):
    def __init__(self, embed_size=256, hidden_size=512, vocab_size=8000, num_heads=8, num_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Embedding for words
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Positional encoding with d_model=hidden_size
        self.positional_encoding = self._generate_positional_encoding(5000, hidden_size)  # max_len=5000
        self.register_buffer('pe', self.positional_encoding)

        # If embed_size != hidden_size, project embeddings to hidden_size
        self.embed_proj = nn.Linear(embed_size, hidden_size) if embed_size != hidden_size else nn.Identity()

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape: (1, max_len, d_model)

    def forward(self, features, captions):
        # features: (batch_size, embed_size)
        # captions: (batch_size, max_seq_length)

        # Prepare target sequences (excluding last token for teacher forcing)
        tgt = captions[:, :-1]  # (batch_size, seq_len-1)
        batch_size, seq_len = tgt.size()

        # Embed the target words
        tgt_emb = self.embedding(tgt) * math.sqrt(self.embed_size)  # (batch_size, seq_len-1, embed_size)
        tgt_emb = self.embed_proj(tgt_emb)  # (batch_size, seq_len-1, hidden_size)

        # Add positional encoding
        # Ensure positional encoding matches hidden_size
        tgt_emb = tgt_emb + self.pe[:, :seq_len, :].to(tgt_emb.device)  # (batch_size, seq_len-1, hidden_size)

        # Dropout
        tgt_emb = self.dropout(tgt_emb)

        # Transformer expects shape: (seq_len, batch_size, d_model)
        tgt_emb = tgt_emb.permute(1, 0, 2)  # (seq_len-1, batch_size, hidden_size)

        # Prepare memory from image features
        # memory shape: (1, batch_size, hidden_size)
        memory = features.unsqueeze(0)  # (1, batch_size, embed_size)
        if self.embed_size != self.hidden_size:
            memory = self.embed_proj(memory)  # (1, batch_size, hidden_size)

        # Generate output
        # We use a subsequent mask for the target to prevent "peeking" at future tokens
        tgt_mask = self._generate_square_subsequent_mask(seq_len).to(tgt_emb.device)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # (seq_len-1, batch_size, hidden_size)
        output = output.permute(1, 0, 2)  # (batch_size, seq_len-1, hidden_size)
        output = self.fc(output)  # (batch_size, seq_len-1, vocab_size)

        return output

    def _generate_square_subsequent_mask(self, sz):
        # Generates a triangular (subsequent) mask for the target sequence
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def sample(self, features, max_len=50, end_token_idx=None):
        # Generate captions using greedy search
        device = features.device
        # memory: (1, batch_size, hidden_size)
        memory = features.unsqueeze(0)  # (1, batch_size, embed_size)
        if self.embed_size != self.hidden_size:
            memory = self.embed_proj(memory)  # (1, batch_size, hidden_size)

        sampled_ids = []
        # Start token is assumed to be <start> with index word2idx["<start>"] = 1
        # If different, change accordingly.
        start_token_idx = 1
        input_ids = torch.tensor([[start_token_idx]], device=device)  # (1, 1)

        for _ in range(max_len):
            tgt_emb = self.embedding(input_ids) * math.sqrt(self.embed_size)  # (1, current_len, embed_size)
            tgt_emb = self.embed_proj(tgt_emb)  # (1, current_len, hidden_size)
            tgt_emb = tgt_emb + self.pe[:, :tgt_emb.size(1), :].to(device)  # (1, current_len, hidden_size)
            tgt_emb = self.dropout(tgt_emb)

            # Transformer expects shape: (current_len, batch_size, hidden_size)
            tgt_emb = tgt_emb.permute(1, 0, 2)  # (current_len, batch_size=1, hidden_size)

            # Generate mask for the current input length
            tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(0)).to(device)

            # Decode
            output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # (current_len, 1, hidden_size)
            output = output.permute(1, 0, 2)  # (1, current_len, hidden_size)
            output = self.fc(output[:, -1, :])  # (1, vocab_size)
            predicted = output.argmax(1)  # (1)

            sampled_ids.append(predicted.item())

            if predicted.item() == end_token_idx:
                break  # Stop if <end> token is generated

            # Append predicted token for next step
            input_ids = torch.cat([input_ids, predicted.unsqueeze(0)], dim=1)  # (1, current_len+1)

        return sampled_ids