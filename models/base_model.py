import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=200): 
        super(EncoderCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.embed = nn.Linear(256, embed_size) 
        self.init_weights()

    def init_weights(self):
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
    def __init__(self, embed_size=200, hidden_size=512, vocab_size=5000, embedding_matrix=None, num_layers=1, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        embeddings = self.dropout(embeddings)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

    def beam_search(self, features, beam_width, word2idx, idx2word, max_len=20):
        sequences = [[[], 1.0]]
        for _ in range(max_len):
            all_candidates = []
            for seq, score in sequences:
                if len(seq) > 0 and seq[-1] == word2idx["<end>"]:
                    all_candidates.append((seq, score))
                    continue
                with torch.no_grad():
                    inputs = torch.LongTensor(seq).view(1, -1).to(features.device)
                    embedded = self.embedding(inputs)
                    lstm_out, _ = self.lstm(torch.cat((features.unsqueeze(1), embedded), 1))
                    logits = self.linear(lstm_out[:, -1, :])
                    probs = F.softmax(logits, dim=1)
                    topk_probs, topk_idx = probs[0].topk(beam_width)

                for i, prob in zip(topk_idx, topk_probs):
                    candidate = [seq + [i.item()], score * prob.item()]
                    all_candidates.append(candidate)
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        best_seq = sequences[0][0]
        return " ".join([idx2word[idx] for idx in best_seq if idx not in [word2idx["<start>"], word2idx["<end>"]]])

    # Implement the sample method
    def sample(self, features, max_len=20, end_token_idx=None):
        """
        Generate a caption for an image given the extracted features.
        Uses greedy search to generate the caption.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        
        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)  # Get the LSTM output and states
            outputs = self.linear(hiddens.squeeze(1))    # Predict the next word
            _, predicted = outputs.max(1)               # Get the word with the highest score
            sampled_ids.append(predicted.item())        # Store the word ID

            # If the predicted word is the end token, stop generating
            if predicted.item() == end_token_idx:
                break

            # Prepare the input for the next time step
            inputs = self.embedding(predicted).unsqueeze(1)
        
        return sampled_ids