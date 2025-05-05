import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from ..general.attention.additive import AdditiveAttention


class NewsEncoder(torch.nn.Module):
    def __init__(self, model_path, num_attention_heads, additive_attn_hidden_dim):
        super(NewsEncoder, self).__init__()

        self.plm_config = AutoConfig.from_pretrained(model_path, cache_dir="/tmp/")
        self.plm = AutoModel.from_config(self.plm_config)
        self.plm.requires_grad_(False)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.plm_config.hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        self.additive_attention = AdditiveAttention(self.plm_config.hidden_size, additive_attn_hidden_dim)

    @property
    def embedding_size(self) -> int:
        return self.plm_config.hidden_size

    def forward(self, news_tokens: torch.Tensor) -> torch.Tensor:
        nonzero = news_tokens.bool()
        padding_mask = ~nonzero

        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        token_embeddings = self.plm(news_tokens, attention_mask=nonzero.int()).last_hidden_state

        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        multihead_attn_output, _ = self.multihead_attention(
            token_embeddings, token_embeddings, token_embeddings, key_padding_mask=padding_mask
        )

        # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        news_vectors, _, _ = self.additive_attention(multihead_attn_output, padding_mask=padding_mask)

        return news_vectors
