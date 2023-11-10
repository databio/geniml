import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim):
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        attention_weights = F.softmax(self.attention(x), dim=1)
        # attention_weights: [batch_size, seq_len, 1]
        weighted_sum = torch.sum(x * attention_weights, dim=1)
        # weighted_sum: [batch_size, embed_dim]
        return weighted_sum
