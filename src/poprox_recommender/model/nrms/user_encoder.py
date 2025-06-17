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

    def forward(self, article_vectors):
        """
        Args:
            news_vectors: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        nonzero_inputs = article_vectors.sum(dim=2).bool()

        # Multi-head attention needs at least one position to be unmasked
        # or it returns NaNs, so unmask the last position even if it's padding
        # (Note: article interaction histories are padded on the left)
        mha_padding_mask = ~nonzero_inputs
        mha_padding_mask[:, -1] = False

        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        multihead_attn_output, _ = self.multihead_attention(
            article_vectors, article_vectors, article_vectors, key_padding_mask=mha_padding_mask
        )

        # Additive attention can handle all positions being masked and will zero
        # the weights of the non-zero vectors coming from MHA in masked positions
        add_padding_mask = ~nonzero_inputs

        # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        user_vectors, _, _ = self.additive_attention(multihead_attn_output, padding_mask=add_padding_mask)

        return user_vectors
