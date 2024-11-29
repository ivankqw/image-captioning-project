import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import maskrcnn_resnet50_fpn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256, device='cuda'):
        super(EncoderCNN, self).__init__()
        self.device = device

        # Load the pre-trained Mask R-CNN model
        self.mask_rcnn = maskrcnn_resnet50_fpn(weights='DEFAULT')
        self.mask_rcnn.to(self.device)
        self.mask_rcnn.eval()

        # Freeze Mask R-CNN parameters
        for param in self.mask_rcnn.parameters():
            param.requires_grad = False

        # Access the backbone, RPN, and ROI Heads from Mask R-CNN
        self.backbone = self.mask_rcnn.backbone
        self.rpn = self.mask_rcnn.rpn
        self.roi_heads = self.mask_rcnn.roi_heads
        self.transform = self.mask_rcnn.transform

        # Global feature embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_embed = nn.Linear(self.backbone.out_channels, embed_size)

        # Object feature embedding
        # The output of the box head is 1024-dimensional
        self.obj_embed = nn.Linear(1024, embed_size)

        # Initialize weights of the linear layers
        self.init_weights()

    def init_weights(self):
        # Initialize weights of the linear layers using Xavier initialization
        nn.init.xavier_uniform_(self.global_embed.weight)
        nn.init.zeros_(self.global_embed.bias)
        nn.init.xavier_uniform_(self.obj_embed.weight)
        nn.init.zeros_(self.obj_embed.bias)

    def train(self, mode=True):
        # Override the train method to prevent Mask R-CNN from switching to train mode
        super(EncoderCNN, self).train(mode)
        self.mask_rcnn.eval()  # Ensure Mask R-CNN stays in eval mode

    def forward(self, images):
        """
        Forward pass through the encoder.
        Args:
            images: Tensor of shape (batch_size, C, H, W)
        Returns:
            combined_features: Tensor of shape (batch_size, embed_size * 2)
        """
        # Convert the batch tensor to a list of individual image tensors
        images = list(images)

        # Use Mask R-CNN's transform to preprocess the images
        transformed_images, _ = self.transform(images)

        # Extract features using the backbone
        with torch.no_grad():
            features = self.backbone(transformed_images.tensors)

        # Global feature extraction
        # Use the highest resolution feature map (assuming key '0')
        feature_map = features['0']  # Shape: (batch_size, C, H, W)
        global_features = self.global_pool(feature_map)  # Shape: (batch_size, C, 1, 1)
        global_features = global_features.view(global_features.size(0), -1)  # Shape: (batch_size, C)
        global_features = self.global_embed(global_features)  # Shape: (batch_size, embed_size)

        # Object detection and feature extraction
        # Get proposals from RPN
        with torch.no_grad():
            proposals, _ = self.rpn(transformed_images, features, None)

        # Get detections from ROI Heads
        detections, _ = self.roi_heads(features, proposals, transformed_images.image_sizes, None)

        # Extract object features using the box head
        object_features_list = []
        for i in range(len(detections)):
            boxes = detections[i]['boxes']  # Bounding boxes for detected objects

            if boxes.shape[0] > 0:
                # Perform RoI pooling on the detected boxes
                box_features = self.roi_heads.box_roi_pool(
                    features, [boxes], [transformed_images.image_sizes[i]]
                )
                # Pass through the box head to get object features
                box_features = self.roi_heads.box_head(box_features)  # Shape: (num_boxes, 1024)
                # Transform to embed_size
                object_features = self.obj_embed(box_features)  # Shape: (num_boxes, embed_size)
                # Aggregate object features by averaging
                aggregated_feats = object_features.mean(dim=0)  # Shape: (embed_size,)
            else:
                # If no objects detected, use a zero vector
                aggregated_feats = torch.zeros(self.obj_embed.out_features).to(self.device)

            object_features_list.append(aggregated_feats)

        # Combine object features into a tensor
        object_features = torch.stack(object_features_list, dim=0)  # Shape: (batch_size, embed_size)

        # Concatenate global and object features
        combined_features = torch.cat([global_features, object_features], dim=1)  # Shape: (batch_size, embed_size * 2)
        return combined_features

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
    
class DecoderRNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, vocab_size, dropout=0.5):
        super(DecoderRNN, self).__init__()
        # Embedding layer to convert word indices to embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Adjusted input size for the feature projection layer
        self.feature_proj = nn.Linear(input_size, embed_size)
        # Custom LSTM cell
        self.lstm_cell = LSTM(embed_size, hidden_size)
        # Fully connected layer to project hidden state to vocabulary space
        self.fc = nn.Linear(hidden_size, vocab_size)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize weights for embedding and fully connected layers
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)

    def forward(self, features, captions):
        """
        Forward pass through the decoder.
        Args:
            features: Combined image features from the encoder, shape (batch_size, input_size)
            captions: Caption sequences, shape (batch_size, max_seq_length)
        Returns:
            outputs: Predicted word distributions, shape (batch_size, seq_len, vocab_size)
        """
        # Project the combined features to embed_size
        features = self.feature_proj(features)  # Shape: (batch_size, embed_size)

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

    def sample(self, features, max_len=20, end_token_idx=None):
        """
        Generate captions for given image features using greedy search.
        Args:
            features: Combined image features from the encoder, shape (1, input_size)
            max_len: Maximum length of the generated caption
            end_token_idx: Index of the <end> token
        Returns:
            sampled_ids: List of predicted word indices
        """
        # Project the combined features to embed_size
        features = self.feature_proj(features)  # Shape: (1, embed_size)

        sampled_ids = []
        inputs = features  # Initial input is the image features
        h_t = torch.zeros(1, self.hidden_size)
        c_t = torch.zeros(1, self.hidden_size)
        
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