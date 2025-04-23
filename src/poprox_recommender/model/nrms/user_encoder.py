import torch
from torch import nn

from poprox_recommender.model.general.attention.additive import (
    AdditiveAttention,
)


class UserEncoder(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(UserEncoder, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(hidden_size, 200)

    def forward(self, news_vectors):
        """
        Args:
            news_vectors: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        nonzero = news_vectors.sum(dim=2).bool()
        padding_mask = ~nonzero

        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        multihead_attn_output, _ = self.multihead_attention(
            news_vectors, news_vectors, news_vectors, key_padding_mask=padding_mask
        )

        # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        user_vectors, _, _ = self.additive_attention(multihead_attn_output, padding_mask=padding_mask)

        return user_vectors
