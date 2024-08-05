import torch
from torch import nn

from poprox_recommender.model.general.attention.additive import (
    AdditiveAttention,
)


class UserEncoder(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(UserEncoder, self).__init__()

        # self.multihead_self_attention = MultiHeadSelfAttention(
        # config.hidden_size, config.num_attention_heads)
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_attention_heads, batch_first=True
        )
        self.additive_attention = AdditiveAttention(hidden_size, 200)

    def forward(self, user_vector):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """

        # batch_size, num_clicked_news_a_user, word_embedding_dim
        # multihead_user_vector = self.multihead_self_attention(user_vector)
        multihead_attn_output, _ = self.multihead_attention(
            user_vector, user_vector, user_vector
        )  # [batch_size, hist_size, emb_dim] -> [batch_size, hist_size, emb_dim]

        # batch_size, word_embedding_dim
        # final_user_vector = self.additive_attention(multihead_user_vector)
        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )  # [batch_size, hist_size, emb_dim] -> [batch_size, emb_dim]

        final_user_vector = torch.sum(additive_attn_output, dim=1)

        return final_user_vector
