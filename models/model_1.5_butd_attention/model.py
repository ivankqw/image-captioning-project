from typing import OrderedDict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2 as fasterrcnn_resnet50_fpn,
)
import torchvision
from torchvision.ops import boxes as box_ops

SCORE_THRESH = 0.2
DETECTIONS_PER_IMG = 36


# adapted from torchvision.models.detection.roi_heads.RoIHeads
def _postprocess_detections(
    roi_heads: nn.Module,
    class_logits: torch.Tensor,  # type: Tensor
    box_regression: torch.Tensor,  # type: Tensor
    proposals: List[torch.Tensor],  # type: List[Tensor]
    image_shapes: List[Tuple[int, int]],
) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]
]:
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    all_indices = []
    start_idx = 0
    for i, (boxes, scores, image_shape, proposal) in enumerate(
        zip(pred_boxes_list, pred_scores_list, image_shapes, proposals)
    ):
        print(f"Processing image {i+1}/{len(image_shapes)}")
        print(f"Initial boxes shape: {boxes.shape}")
        print(f"Initial scores shape: {scores.shape}")
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        print(f"Boxes shape after clipping: {boxes.shape}")

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]
        print(
            f"Shapes after removing background: boxes {boxes.shape}, scores {scores.shape}, labels {labels.shape}"
        )

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        print(
            f"Shapes after reshaping: boxes {boxes.shape}, scores {scores.shape}, labels {labels.shape}"
        )

        # remove low scoring boxes
        inds = torch.where(scores > SCORE_THRESH)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
        print(
            f"Shapes after score thresholding: boxes {boxes.shape}, scores {scores.shape}, labels {labels.shape}"
        )

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        print(
            f"Shapes after removing small boxes: boxes {boxes.shape}, scores {scores.shape}, labels {labels.shape}"
        )

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, roi_heads.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[:DETECTIONS_PER_IMG]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        print(
            f"Shapes after NMS: boxes {boxes.shape}, scores {scores.shape}, labels {labels.shape}"
        )

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

        adjusted_inds = inds[keep] + start_idx
        all_indices.append(adjusted_inds)
        print(f"Adjusted indices: {adjusted_inds}")
        # print(f"Start index: {start_idx}, Max adjusted index: {adjusted_inds.max().item()}")

        start_idx += 1000

    return all_boxes, all_scores, all_labels, all_indices


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
        print("Batch size:", batch_size)

        # Extract bottom-up features for the batch
        transformed_images, _ = self.faster_rcnn.transform(images)
        features = self.faster_rcnn.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.faster_rcnn.rpn(transformed_images, features, None)
        box_features = self.faster_rcnn.roi_heads.box_roi_pool(
            features, proposals, transformed_images.image_sizes
        )
        box_features = self.faster_rcnn.roi_heads.box_head(box_features)
        print("Shape of box_features:", box_features.shape)
        # box prediction
        class_logits, box_regression = self.faster_rcnn.roi_heads.box_predictor(
            box_features
        )
        # adapted postprocess detections to get the indices of the boxes that we want to keep thus box features
        _, _, _, indices_keep = _postprocess_detections(
            self.faster_rcnn.roi_heads,
            class_logits,
            box_regression,
            proposals,
            transformed_images.image_sizes,
        )
        indices_keep = torch.cat(indices_keep, dim=0)
        print("Indices of boxes to keep:", indices_keep)
        # keep the box_features of the kept proposals
        box_features = box_features[indices_keep]
        print("Shape of box_features after filtering:", box_features.shape)

        return box_features


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
        att1 = self.features_att(image_features)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        features_dim=1024,
        dropout=0.5,
        device="cuda",
    ):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(
            features_dim, decoder_dim, attention_dim
        )  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(
            embed_dim + features_dim + decoder_dim, decoder_dim, bias=True
        )  # top down attention LSTMCell
        self.language_model = nn.LSTMCell(
            features_dim + decoder_dim, decoder_dim, bias=True
        )  # language model LSTMCell
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.fc = weight_norm(
            nn.Linear(decoder_dim, vocab_size)
        )  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        self.device = device

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.decoder_dim).to(
            self.device
        )  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim).to(self.device)
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = image_features.size(0)
        vocab_size = self.vocab_size

        # Print shapes of input tensors
        print("Shape of image_features:", image_features.shape)
        print("Shape of encoded_captions:", encoded_captions.shape)
        print("Shape of caption_lengths:", caption_lengths.shape)

        # Flatten image
        image_features_mean = image_features.mean(1).to(
            self.device
        )  # (batch_size, num_pixels, encoder_dim)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Print shapes after sorting
        print("Shape of image_features after sorting:", image_features.shape)
        print("Shape of image_features_mean after sorting:", image_features_mean.shape)
        print("Shape of encoded_captions after sorting:", encoded_captions.shape)
        print("Shape of caption_lengths after sorting:", caption_lengths.shape)

        # Embedding
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h1, c1 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(batch_size)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            self.device
        )
        predictions1 = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            self.device
        )

        # At each time-step, pass the language model's previous hidden state, the mean pooled bottom up features and
        # word embeddings to the top down attention model. Then pass the hidden state of the top down model and the bottom up
        # features to the attention block. The attention weighed bottom up features and hidden state of the top down attention model
        # are then passed to the language model
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            h1, c1 = self.top_down_attention(
                torch.cat(
                    [
                        h2[:batch_size_t],
                        image_features_mean[:batch_size_t],
                        embeddings[:batch_size_t, t, :],
                    ],
                    dim=1,
                ),
                (h1[:batch_size_t], c1[:batch_size_t]),
            )
            attention_weighted_encoding = self.attention(
                image_features[:batch_size_t], h1[:batch_size_t]
            )
            preds1 = self.fc1(self.dropout(h1))
            h2, c2 = self.language_model(
                torch.cat(
                    [attention_weighted_encoding[:batch_size_t], h1[:batch_size_t]],
                    dim=1,
                ),
                (h2[:batch_size_t], c2[:batch_size_t]),
            )
            preds = self.fc(self.dropout(h2))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            predictions1[:batch_size_t, t, :] = preds1

        return predictions, predictions1, encoded_captions, decode_lengths, sort_ind

    def sample(self, features, word_map, max_len=20, end_token_idx=None):
        """
        Generate captions for given image features using greedy search.
        Args:
            features: Image features from the encoder, shape (1, embed_size)
            word_map: Word map dictionary
            rev_word_map: Reverse word map dictionary
            max_len: Maximum length of the generated caption
            end_token_idx: Index of the <end> token
        Returns:
            sampled_ids: List of predicted word indices
        """
        sampled_ids = []
        inputs = features  # Initial input is the image features
        h_t = torch.zeros(1, self.decoder_dim).to(self.device)
        c_t = torch.zeros(1, self.decoder_dim).to(self.device)

        for _ in range(max_len):
            embeddings = self.embedding(
                torch.tensor([word_map["<start>"]], device=self.device)
            ).squeeze(
                1
            )  # (1, embed_dim)
            h1, c1 = self.top_down_attention(
                torch.cat([h_t, inputs, embeddings], dim=1),
                (h_t, c_t),
            )
            attention_weighted_encoding, _ = self.attention(inputs, h1)
            h_t, c_t = self.language_model(
                torch.cat([attention_weighted_encoding, h1], dim=1),
                (h_t, c_t),
            )
            outputs = self.fc(h_t)  # Compute word distribution
            predicted = outputs.argmax(1)  # Get the index with the highest probability
            sampled_ids.append(predicted.item())

            if predicted.item() == end_token_idx:
                break  # Stop if <end> token is generated

            # Prepare input for next time step
            inputs = self.embedding(predicted)

        return sampled_ids

    def sample_beam(self, features, word_map, rev_word_map, beam_size=5, max_len=20):
        """
        Generate captions for given image features using beam search.
        Args:
            features: Image features from the encoder, shape (1, embed_size)
            word_map: Word map dictionary
            rev_word_map: Reverse word map dictionary
            beam_size: Beam size for beam search
            max_len: Maximum length of the generated caption
        Returns:
            sampled_ids: List of predicted word indices
        """
        k = beam_size
        vocab_size = self.vocab_size

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map["<start>"]]] * k).to(
            self.device
        )  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1, c1 = self.init_hidden_state(k)  # (batch_size, decoder_dim)
        h2, c2 = self.init_hidden_state(k)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            h1, c1 = self.top_down_attention(
                torch.cat(
                    [h2, features.expand(k, self.features_dim), embeddings], dim=1
                ),
                (h1, c1),
            )  # (batch_size_t, decoder_dim)
            attention_weighted_encoding = self.attention(
                features.expand(k, self.features_dim), h1
            )
            h2, c2 = self.language_model(
                torch.cat([attention_weighted_encoding, h1], dim=1), (h2, c2)
            )

            scores = self.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(
                    k, 0, True, True
                )  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat(
                [seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1
            )  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [
                ind
                for ind, next_word in enumerate(next_word_inds)
                if next_word != word_map["<end>"]
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # Hypotheses
        hypothesis = [
            rev_word_map[w]
            for w in seq
            if w not in {word_map["<start>"], word_map["<end>"], word_map["<pad>"]}
        ]
        hypothesis = " ".join(hypothesis)

        return hypothesis
