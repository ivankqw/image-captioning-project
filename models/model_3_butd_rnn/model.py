from typing import OrderedDict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2 as fasterrcnn_resnet50_fpn,
)

from torchvision.ops import boxes as box_ops

SCORE_THRESH = 0.2
DETECTIONS_PER_IMG = 36


# adapted from torchvision.models.detection.roi_heads.RoIHeads
def _postprocess_detections_single_image(
    roi_heads: nn.Module,
    class_logits: torch.Tensor,  # type: Tensor
    box_regression: torch.Tensor,  # type: Tensor
    proposals: List[torch.Tensor],  # type: List[Tensor]
    image_shape: Tuple[int, int],
) -> List[int]:

    pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)
    pred_scores = F.softmax(class_logits, -1)

    # remove predictions with the background label
    boxes = pred_boxes[:, 1:]
    scores = pred_scores[:, 1:]

    # batch everything, by making every class prediction be a separate instance
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1)

    # remove low scoring boxes
    inds = torch.where(scores > roi_heads.score_thresh)[0]
    boxes, scores = boxes[inds], scores[inds]

    # remove empty boxes
    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
    boxes, scores = boxes[keep], scores[keep]

    # non-maximum suppression
    keep = box_ops.nms(boxes, scores, SCORE_THRESH)
    # keep only top DETECTIONS_PER_IMG scoring predictions
    keep = keep[:DETECTIONS_PER_IMG]
    boxes, scores = boxes[keep], scores[keep]

    return keep.tolist()


class EncoderBUAttention(nn.Module):
    def __init__(self, device="cuda"):
        super(EncoderBUAttention, self).__init__()
        self.device = device

        # Load the pre-trained Faster R-CNN
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights="COCO_V1")
        self.faster_rcnn.to(self.device)
        self.faster_rcnn.eval()

        # freeze the model
        for param in self.faster_rcnn.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        # override the train method to prevent the model from being set to training mode
        # this is because the Faster R-CNN model is already in eval mode
        super(EncoderBUAttention, self).train(mode)
        self.faster_rcnn.eval()  # set the Faster R-CNN model to eval mode

    def forward(self, images):
        batch_size = len(images)

        # Extract bottom-up features for the batch
        transformed_images, _ = self.faster_rcnn.transform(images)
        features = self.faster_rcnn.backbone(transformed_images.tensors.to(self.device))
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals, _ = self.faster_rcnn.rpn(transformed_images, features, None)

        # extract box_features for each image
        box_features_list = []
        for image, proposals_per_image, image_shape in zip(
            images, proposals, transformed_images.image_sizes
        ):
            box_features = self.faster_rcnn.roi_heads.box_roi_pool(
                features, [proposals_per_image], [image_shape]
            )
            box_features = self.faster_rcnn.roi_heads.box_head(box_features)

            # box prediction
            class_logits, box_regression = self.faster_rcnn.roi_heads.box_predictor(
                box_features
            )

            # adapted postprocess detections to get the indices of the boxes that we want to keep thus box features
            indices_keep = _postprocess_detections_single_image(
                self.faster_rcnn.roi_heads,
                class_logits,
                box_regression,
                [proposals_per_image],
                image_shape,
            )

            # keep the box_features of the kept proposals
            box_features = box_features[indices_keep]

            # Pad box_features with zeros to ensure a consistent shape
            pad_size = DETECTIONS_PER_IMG - box_features.shape[0]
            padded_box_features = torch.cat(
                [
                    box_features,
                    torch.zeros(pad_size, box_features.shape[1], device=self.device),
                ],
                dim=0,
            )
            box_features_list.append(padded_box_features)

        # stack the padded box_features tensors along the batch dimension
        box_features_tensor = torch.stack(box_features_list, dim=0)
        return (
            box_features_tensor  # shape (batch_size, DETECTIONS_PER_IMG, feature_dim)
        )


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
    def __init__(
        self,
        embed_size=256,
        hidden_size=512,
        vocab_size=5000,
        dropout=0.5,
        features_dim=1024,
    ):
        super(DecoderRNN, self).__init__()
        # Embedding layer to convert word indices to embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # Linear layer to transform bottom-up attention features
        self.features_transform = nn.Linear(features_dim, embed_size)

        # Custom LSTM cell (modified input size to handle mean pooled features)
        self.lstm_cell = LSTM(
            embed_size * 2, hidden_size
        )  # * 2 because we concatenate embeddings and features

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
        nn.init.uniform_(self.features_transform.weight, -0.1, 0.1)
        nn.init.constant_(self.features_transform.bias, 0)

    def forward(self, features, captions):
        """
        Forward pass through the decoder.
        Args:
            features: Image features from the encoder, shape (batch_size, num_regions, features_dim)
            captions: Caption sequences, shape (batch_size, max_seq_length)
        Returns:
            outputs: Predicted word distributions, shape (batch_size, seq_len, vocab_size)
        """
        # Mean pool the features across regions
        features_mean = features.mean(1)  # Shape: (batch_size, features_dim)

        # Transform features to embed_size dimension
        features_embedded = self.features_transform(
            features_mean
        )  # Shape: (batch_size, embed_size)

        # Embed the captions (exclude the last word for teacher forcing)
        embeddings = self.embedding(
            captions[:, :-1]
        )  # Shape: (batch_size, seq_len - 1, embed_size)

        batch_size, seq_len, _ = embeddings.size()
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(
            features.device
        )

        # Initialize hidden and cell states
        h_t = torch.zeros(batch_size, self.hidden_size).to(features.device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(features.device)

        # Unroll the LSTM for seq_len time steps
        for t in range(seq_len):
            # Concatenate embedding with features at each time step
            x_t = torch.cat([embeddings[:, t, :], features_embedded], dim=1)
            x_t = self.dropout(x_t)

            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)
            output = self.fc(h_t)
            outputs[:, t, :] = output

        return outputs

    def sample(self, features, max_len=20, end_token_idx=None):
        """
        Generate captions for given image features using greedy search.
        Args:
            features: Image features from the encoder, shape (1, num_regions, features_dim)
            max_len: Maximum length of the generated caption
            end_token_idx: Index of the <end> token
        Returns:
            sampled_ids: List of predicted word indices
        """
        sampled_ids = []

        # Mean pool the features across regions
        features_mean = features.mean(1)  # Shape: (1, features_dim)
        features_embedded = self.features_transform(
            features_mean
        )  # Shape: (1, embed_size)

        # Initialize LSTM states
        h_t = torch.zeros(1, self.hidden_size).to(features.device)
        c_t = torch.zeros(1, self.hidden_size).to(features.device)

        # Initialize first input with <start> token
        word_embed = self.embedding(
            torch.tensor([1]).to(features.device)
        )  # Assuming 1 is <start> token

        for _ in range(max_len):
            # Concatenate word embedding with features
            x_t = torch.cat([word_embed, features_embedded], dim=1)
            x_t = self.dropout(x_t)

            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)
            outputs = self.fc(h_t)
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())

            if predicted.item() == end_token_idx:
                break

            # Prepare input for next time step
            word_embed = self.embedding(predicted)

        return sampled_ids
