from collections import OrderedDict  # Corrected import for OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2 as fasterrcnn_resnet50_fpn,
)
from torchvision.ops import boxes as box_ops

SCORE_THRESH = 0.2  # Used for score thresholding
NMS_THRESH = 0.5  # Non-maximum suppression threshold
DETECTIONS_PER_IMG = 36  # Max detections per image


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


class Attention(nn.Module):
    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features, decoder_hidden):
        # Compute attention weights
        att1 = self.features_att(
            image_features
        )  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(
            2
        )  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(
            dim=1
        )  # (batch_size, features_dim)
        return attention_weighted_encoding


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)  # Shape: (max_len, 1, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, embed_dim)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class DecoderWithTransformer(nn.Module):
    """
    Decoder with Transformer.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        num_layers,
        num_heads,
        features_dim=1024,
        dropout=0.5,
        device="cuda",
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's transformer
        :param vocab_size: size of vocabulary
        :param num_layers: number of transformer layers
        :param num_heads: number of attention heads
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithTransformer, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = Attention(
            features_dim, embed_dim, attention_dim
        )  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout).to(
            device
        )
        decoder_layer = nn.TransformerDecoderLayer(
            embed_dim, num_heads, decoder_dim, dropout
        ).to(device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers).to(
            device
        )
        self.fc = weight_norm(nn.Linear(embed_dim, vocab_size)).to(device)
        self.encoder_projection = nn.Linear(features_dim, embed_dim).to(
            device
        )  # Project features_dim to embed_dim
        self.init_weights()
        self.device = device

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.device)) == 1
        mask = (
            mask.float().masked_fill(~mask, float("-inf")).masked_fill(mask, float(0.0))
        )
        return mask  # Shape: (sz, sz)

    def forward(self, image_features, encoded_captions, caption_lengths):
        # Sort input data by decreasing caption lengths
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        # Prepare decode_lengths
        decode_lengths = (caption_lengths - 1).tolist()

        # Permute for transformer input
        embeddings = embeddings.permute(
            1, 0, 2
        )  # (max_caption_length, batch_size, embed_dim)
        embeddings = self.positional_encoding(embeddings)

        # Generate the attention mask
        max_len = embeddings.size(0)
        tgt_mask = self.generate_square_subsequent_mask(max_len)

        # Attention: get context vectors
        image_features_mean = image_features.mean(1)  # (batch_size, features_dim)
        attention_weighted_encoding = self.attention(
            image_features, image_features_mean
        )  # (batch_size, features_dim)

        # Project encoder output to embed_dim
        memory = self.encoder_projection(attention_weighted_encoding)
        memory = memory.unsqueeze(0)  # (1, batch_size, embed_dim)

        # Transformer Decoder
        transformer_output = self.transformer_decoder(
            embeddings, memory, tgt_mask=tgt_mask
        )  # (max_caption_length, batch_size, embed_dim)

        # Final output
        outputs = self.fc(
            transformer_output
        )  # (max_caption_length, batch_size, vocab_size)
        outputs = outputs.permute(
            1, 0, 2
        )  # (batch_size, max_caption_length, vocab_size)

        return outputs, encoded_captions, decode_lengths

    def sample(self, features, word_map, max_len=20, end_token_idx=None):
        """
        Generate captions for given image features using greedy search.
        Args:
            features: Image features from the encoder, shape (1, DETECTIONS_PER_IMG, features_dim)
            word_map: Word map dictionary
            max_len: Maximum length of the generated caption
            end_token_idx: Index of the <end> token
        Returns:
            sampled_ids: List of predicted word indices
        """
        self.eval()
        with torch.no_grad():
            inputs = features.to(self.device)

            # Attention
            image_features_mean = inputs.mean(1)
            attention_weighted_encoding = self.attention(
                inputs, image_features_mean
            )  # (batch_size, features_dim)
            memory = self.encoder_projection(attention_weighted_encoding)
            memory = memory.unsqueeze(0)  # Shape: (1, batch_size, embed_dim)

            # Initialize input token (start token)
            sampled_ids = torch.LongTensor([word_map["<start>"]]).to(self.device)
            sampled_ids = sampled_ids.unsqueeze(1)  # Shape: (batch_size, 1)

            for _ in range(max_len):
                embeddings = self.embedding(
                    sampled_ids
                )  # (batch_size, seq_len, embed_dim)
                embeddings = embeddings.permute(
                    1, 0, 2
                )  # (seq_len, batch_size, embed_dim)
                embeddings = self.positional_encoding(embeddings)
                tgt_mask = self.generate_square_subsequent_mask(embeddings.size(0))

                transformer_output = self.transformer_decoder(
                    embeddings, memory, tgt_mask=tgt_mask
                )
                outputs = self.fc(
                    transformer_output
                )  # (seq_len, batch_size, vocab_size)
                outputs = outputs[
                    -1, :, :
                ]  # Get last time step (batch_size, vocab_size)
                predicted = outputs.argmax(1)  # (batch_size)

                sampled_ids = torch.cat(
                    [sampled_ids, predicted.unsqueeze(1)], dim=1
                )  # (batch_size, seq_len + 1)

                if predicted.item() == end_token_idx:
                    break

            sampled_ids = sampled_ids.squeeze().tolist()
            return sampled_ids

    def sample_beam(
        self,
        features,
        word_map,
        rev_word_map,
        beam_size=5,
        max_len=20,
        end_token_idx=None,
    ):
        """
        Generate captions for given image features using beam search.
        Args:
            features: Image features from the encoder, shape (1, DETECTIONS_PER_IMG, features_dim)
            word_map: Word map dictionary
            rev_word_map: Reverse word map dictionary
            beam_size: Beam size for beam search
            max_len: Maximum length of the generated caption
            end_token_idx: Index of the <end> token
        Returns:
            hypothesis: Generated caption as a string.
        """
        # Feature preparation
        inputs = features.to(self.device)  # (1, DETECTIONS_PER_IMG, features_dim)

        # Attention
        image_features_mean = inputs.mean(1)
        attention_weighted_encoding = self.attention(
            inputs, image_features_mean
        )  # (1, features_dim)
        encoder_output = self.encoder_projection(attention_weighted_encoding)
        encoder_output = encoder_output.unsqueeze(0)  # (1, 1, embed_dim)

        # Start the beam search
        k = beam_size
        vocab_size = self.vocab_size

        # Initialize sequences
        seqs = torch.LongTensor([[word_map["<start>"]]] * k).to(self.device)  # (k, 1)
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

        complete_seqs = []
        complete_seqs_scores = []

        # Expand encoder output
        encoder_output = encoder_output.expand(-1, k, -1)  # (1, k, embed_dim)

        for step in range(max_len):
            embeddings = self.embedding(seqs)  # (k, seq_len, embed_dim)
            embeddings = embeddings.permute(1, 0, 2)  # (seq_len, k, embed_dim)
            embeddings = self.positional_encoding(embeddings)
            tgt_mask = self.generate_square_subsequent_mask(embeddings.size(0))

            decoder_outputs = self.transformer_decoder(
                embeddings, encoder_output, tgt_mask=tgt_mask
            )  # (seq_len, k, embed_dim)
            outputs = self.fc(decoder_outputs[-1])  # (k, vocab_size)
            scores = F.log_softmax(outputs, dim=1)  # (k, vocab_size)

            # Add scores to previous scores
            scores = top_k_scores.expand_as(scores) + scores  # (k, vocab_size)

            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                # Flatten scores to (k * vocab_size)
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

            prev_word_inds = top_k_words // vocab_size  # (k)
            next_word_inds = top_k_words % vocab_size  # (k)

            # Update sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
            )  # (k, step+1)

            # Check for complete sequences
            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != end_token_idx
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # Reduce beam size

            if k == 0:
                break

            # Proceed with incomplete sequences
            seqs = seqs[incomplete_inds]
            encoder_output = encoder_output[:, incomplete_inds, :]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

        # Select the best sequence
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # Convert indices to words
        hypothesis = [
            rev_word_map[w]
            for w in seq
            if w not in {word_map["<start>"], word_map["<end>"], word_map["<pad>"]}
        ]
        hypothesis = " ".join(hypothesis)
        return hypothesis
