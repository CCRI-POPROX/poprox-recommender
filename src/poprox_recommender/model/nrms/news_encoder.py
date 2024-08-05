import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from poprox_recommender.model.general.attention.newsadditive import (
    NewsAdditiveAttention,
)
from poprox_recommender.paths import model_file_path


class NewsEncoder(torch.nn.Module):
    def __init__(self, pretrained_model, num_attention_heads, additive_attn_hidden_dim):
        super(NewsEncoder, self).__init__()

        self.plm = AutoModel.from_pretrained(model_file_path(pretrained_model), cache_dir="/tmp/")

        self.plm_hidden_size = AutoConfig.from_pretrained(
            model_file_path(pretrained_model), cache_dir="/tmp/"
        ).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.plm_hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        self.additive_attention = NewsAdditiveAttention(self.plm_hidden_size, additive_attn_hidden_dim)

    def forward(self, news_input: torch.Tensor) -> torch.Tensor:
        # batch_size, num_words_title, word_embedding_dim

        V = self.plm(news_input).last_hidden_state
        multihead_attn_output, _ = self.multihead_attention(
            V, V, V
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]

        additive_attn_output = self.additive_attention(
            multihead_attn_output
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]

        output = torch.sum(
            additive_attn_output, dim=1
        )  # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]

        return output
