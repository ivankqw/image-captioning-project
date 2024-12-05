import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection import maskrcnn_resnet50_fpn

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256, device='cuda', top_k=5):
        """
        Encoder using Mask R-CNN to extract global and object-level features.

        Args:
            embed_size (int): Dimension of the embedding vectors.
            device (str): Device to run the model on ('cuda' or 'cpu').
            top_k (int): Number of top objects to consider based on confidence scores.
        """
        super(EncoderCNN, self).__init__()
        self.device = device
        self.top_k = top_k  # Number of top objects to consider

        # Load the pre-trained Mask R-CNN model
        self.mask_rcnn = maskrcnn_resnet50_fpn(weights='DEFAULT')
        self.mask_rcnn.to(self.device)
        self.mask_rcnn.eval()  # Keep in eval mode

        # Freeze Mask R-CNN parameters to prevent training
        for param in self.mask_rcnn.parameters():
            param.requires_grad = False

        # Access the backbone, RPN, and ROI Heads from Mask R-CNN
        self.backbone = self.mask_rcnn.backbone
        self.rpn = self.mask_rcnn.rpn
        self.roi_heads = self.mask_rcnn.roi_heads
        self.transform = self.mask_rcnn.transform

        # Global feature embedding
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Calculate the total feature dimension after concatenation
        # For ResNet50-FPN, FPN produces 5 feature maps each with 256 channels
        num_feature_maps = 5
        self.global_feature_dim = self.backbone.out_channels * num_feature_maps  # 256 * 5 = 1280
        self.global_embed = nn.Linear(self.global_feature_dim, embed_size)

        # Object feature embedding
        self.obj_embed = nn.Linear(1024, embed_size)

        # Projection for bounding box coordinates (x1, y1, x2, y2)
        self.box_proj = nn.Linear(4, embed_size)

        # Combine object features and box coordinates
        self.obj_comb_proj = nn.Linear(embed_size * 2, embed_size)

        # Fusion layer to combine global and object features
        self.fusion_fc = nn.Linear(embed_size * 2, embed_size)
        self.relu = nn.ReLU()

        # Initialize weights of the linear layers
        self.init_weights()

    def init_weights(self):
        """Initialize weights of the linear layers using Xavier uniform and zeros for biases."""
        nn.init.xavier_uniform_(self.global_embed.weight)
        nn.init.zeros_(self.global_embed.bias)
        nn.init.xavier_uniform_(self.obj_embed.weight)
        nn.init.zeros_(self.obj_embed.bias)
        nn.init.xavier_uniform_(self.box_proj.weight)
        nn.init.zeros_(self.box_proj.bias)
        nn.init.xavier_uniform_(self.obj_comb_proj.weight)
        nn.init.zeros_(self.obj_comb_proj.bias)
        nn.init.xavier_uniform_(self.fusion_fc.weight)
        nn.init.zeros_(self.fusion_fc.bias)

    def train(self, mode=True):
        """
        Override the train method to keep Mask R-CNN in eval mode.

        Args:
            mode (bool): Whether to set to train mode or not.
        """
        super(EncoderCNN, self).train(mode)
        self.mask_rcnn.eval()  # Ensure Mask R-CNN stays in eval mode

    def forward(self, images):
        """
        Forward pass through the encoder.

        Args:
            images (Tensor): Batch of images, shape (batch_size, C, H, W).

        Returns:
            combined_features (Tensor): Combined global and object features, shape (batch_size, embed_size).
            object_features (Tensor): Object features, shape (batch_size, max_objects, embed_size).
        """
        # Convert the batch tensor to a list of individual image tensors
        images = list(images)

        # Use Mask R-CNN's transform to preprocess the images
        transformed_images, _ = self.transform(images)

        # Extract features using the backbone
        with torch.no_grad():
            features = self.backbone(transformed_images.tensors)

        # Global feature extraction
        # Use multiple feature maps and concatenate them
        feature_maps = []
        for f in features.values():
            pooled = self.global_pool(f)  # Shape: (batch_size, C, 1, 1)
            pooled = pooled.view(pooled.size(0), -1)  # Shape: (batch_size, C)
            feature_maps.append(pooled)
        global_features = torch.cat(feature_maps, dim=1)  # Shape: (batch_size, total_C)
        global_features = self.global_embed(global_features)  # Shape: (batch_size, embed_size)

        # Object detection and feature extraction
        # Get proposals from RPN
        with torch.no_grad():
            proposals, _ = self.rpn(transformed_images, features, None)

        # Get detections from ROI Heads
        with torch.no_grad():
            detections, _ = self.roi_heads(features, proposals, transformed_images.image_sizes, None)

        # Extract object features using the box head
        object_features_list = []
        for i in range(len(detections)):
            boxes = detections[i]['boxes']    # Shape: (num_boxes, 4)
            scores = detections[i]['scores']  # Shape: (num_boxes,)

            if boxes.shape[0] > 0:
                # Select top-K boxes based on scores
                topk = min(self.top_k, boxes.shape[0])
                topk_scores, topk_indices = scores.topk(topk)
                topk_boxes = boxes[topk_indices]

                # Perform RoI pooling on the selected boxes
                box_features = self.roi_heads.box_roi_pool(
                    features, [topk_boxes], [transformed_images.image_sizes[i]]
                )
                box_features = self.roi_heads.box_head(box_features)  # Shape: (topk, 1024)

                # Transform to embed_size
                object_features = self.obj_embed(box_features)  # Shape: (topk, embed_size)

                # Project bounding box coordinates
                # Normalize box coordinates by image size
                img_width, img_height = transformed_images.image_sizes[i]
                normalized_boxes = topk_boxes.clone()
                normalized_boxes[:, [0, 2]] /= img_width
                normalized_boxes[:, [1, 3]] /= img_height
                box_coords = self.box_proj(normalized_boxes)  # Shape: (topk, embed_size)

                # Concatenate object features with box coordinates
                combined_obj_feats = torch.cat((object_features, box_coords), dim=1)  # Shape: (topk, embed_size * 2)

                # Project concatenated features back to embed_size
                combined_obj_feats = self.obj_comb_proj(combined_obj_feats)  # Shape: (topk, embed_size)

                object_features_list.append(combined_obj_feats)  # Keep as list of tensors
            else:
                # If no objects detected, use zeros
                combined_obj_feats = torch.zeros((1, self.obj_comb_proj.out_features)).to(self.device)
                object_features_list.append(combined_obj_feats)

        # Stack object features into a tensor
        max_objects = max([f.size(0) for f in object_features_list])
        padded_obj_feats = []
        for feats in object_features_list:
            pad_size = max_objects - feats.size(0)
            if pad_size > 0:
                padding = torch.zeros((pad_size, feats.size(1))).to(self.device)
                feats = torch.cat((feats, padding), dim=0)
            padded_obj_feats.append(feats.unsqueeze(0))  # Shape: (1, max_objects, embed_size)
        object_features = torch.cat(padded_obj_feats, dim=0)  # Shape: (batch_size, max_objects, embed_size)

        # Combine global and object features
        combined = torch.cat([global_features, object_features.mean(dim=1)], dim=1)  # Shape: (batch_size, embed_size * 2)

        # Pass through fusion layer with ReLU activation
        combined_features = self.relu(self.fusion_fc(combined))  # Shape: (batch_size, embed_size)

        return combined_features, object_features

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
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Feature projection layer
        self.feature_proj = nn.Linear(input_size, embed_size)

        # Initialize hidden and cell state from features
        self.init_hidden = nn.Linear(embed_size * 2, hidden_size)
        self.init_cell = nn.Linear(embed_size * 2, hidden_size)

        # LSTM layer using nn.LSTMCell
        self.lstm = nn.LSTMCell(embed_size + embed_size * 2, hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)
        nn.init.xavier_uniform_(self.init_hidden.weight)
        nn.init.zeros_(self.init_hidden.bias)
        nn.init.xavier_uniform_(self.init_cell.weight)
        nn.init.zeros_(self.init_cell.bias)

    def forward(self, global_features, object_features, captions):
        """
        Forward pass through the decoder.

        Args:
            global_features (Tensor): Combined image features from the encoder, shape (batch_size, input_size).
            object_features (Tensor): Object features, shape (batch_size, max_objects, embed_size).
            captions (Tensor): Caption sequences, shape (batch_size, max_seq_length).

        Returns:
            outputs (Tensor): Predicted word distributions, shape (batch_size, seq_len, vocab_size).
        """
        batch_size = global_features.size(0)
        device = global_features.device

        # Project the combined features to embed_size
        features = self.feature_proj(global_features)  # Shape: (batch_size, embed_size)

        # Compute the mean of object features
        object_features_mean = object_features.mean(dim=1)  # Shape: (batch_size, embed_size)

        # Concatenate global features and object features
        combined_features = torch.cat((features, object_features_mean), dim=1)  # Shape: (batch_size, embed_size * 2)

        # Embed the captions (exclude the last word for teacher forcing)
        embeddings = self.embedding(captions[:, :-1])  # Shape: (batch_size, seq_len - 1, embed_size)
        embeddings = self.dropout(embeddings)

        # Initialize hidden and cell states based on combined features
        h_t = self.init_hidden(combined_features)  # Shape: (batch_size, hidden_size)
        c_t = self.init_cell(combined_features)    # Shape: (batch_size, hidden_size)

        # Prepare inputs by concatenating embeddings with combined features
        combined_features_expanded = combined_features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        lstm_inputs = torch.cat((embeddings, combined_features_expanded), dim=2)  # Shape: (batch_size, seq_len - 1, embed_size + embed_size * 2)

        # Unroll the LSTM
        outputs = []
        for t in range(lstm_inputs.size(1)):
            x_t = lstm_inputs[:, t, :]  # Input at time step t
            h_t, c_t = self.lstm(x_t, (h_t, c_t))  # Update hidden and cell states
            output = self.fc(h_t)  # Shape: (batch_size, vocab_size)
            outputs.append(output.unsqueeze(1))  # Append output for each time step

        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, seq_len - 1, vocab_size)
        return outputs

    def sample(self, global_features, object_features, start_token_idx, max_len=20, end_token_idx=None):
        """
        Generate captions for given image features using greedy search.

        Args:
            global_features (Tensor): Combined image features from the encoder, shape (batch_size, embed_size).
            object_features (Tensor): Object features, shape (batch_size, max_objects, embed_size).
            start_token_idx (int): Index of the <start> token.
            max_len (int): Maximum length of the generated caption.
            end_token_idx (int): Index of the <end> token.

        Returns:
            sampled_ids (List[int]): List of predicted word indices.
        """
        device = global_features.device

        # Project the combined features to embed_size
        features = self.feature_proj(global_features)  # Shape: (batch_size, embed_size)

        # Compute the mean of object features
        object_features_mean = object_features.mean(dim=1)  # Shape: (batch_size, embed_size)

        # Concatenate global features and object features
        combined_features = torch.cat((features, object_features_mean), dim=1)  # Shape: (batch_size, embed_size * 2)

        # Initialize hidden and cell states based on combined features
        h_t = self.init_hidden(combined_features)  # Shape: (batch_size, hidden_size)
        c_t = self.init_cell(combined_features)    # Shape: (batch_size, hidden_size)

        # Initialize input with <start> token
        inputs = torch.tensor([start_token_idx], dtype=torch.long).to(device)  # Shape: (1,)

        sampled_ids = []

        for _ in range(max_len):
            # Embed the current input
            embed = self.embedding(inputs).squeeze(0)  # Shape: (embed_size,)
            embed = self.dropout(embed)

            # Concatenate embedding with combined features
            lstm_input = torch.cat((embed, combined_features.squeeze(0)), dim=0)  # Shape: (embed_size + embed_size * 2,)

            # Update hidden and cell states
            h_t, c_t = self.lstm(lstm_input.unsqueeze(0), (h_t, c_t))  # Shape: (1, hidden_size)

            # Compute output word distribution
            output = self.fc(h_t.squeeze(0))  # Shape: (vocab_size,)
            predicted = output.argmax(0)  # Get the index of the max log-probability
            sampled_ids.append(predicted.item())

            if predicted.item() == end_token_idx:
                break  # Stop if <end> token is generated

            # Prepare input for next time step
            inputs = predicted.unsqueeze(0)  # Shape: (1,)

        return sampled_ids