import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from poprox_recommender.paths import model_file_path

from ..general.attention.additive import AdditiveAttention


class NewsEncoder(torch.nn.Module):
    def __init__(self, model_name, num_attention_heads, additive_attn_hidden_dim):
        super(NewsEncoder, self).__init__()

        model_path = model_file_path(model_name)
        self.plm_config = AutoConfig.from_pretrained(model_path, cache_dir="/tmp/")
        self.plm = AutoModel.from_config(self.plm_config)
        self.plm.requires_grad_(False)
        self.plm.eval()

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=self.plm_config.hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        self.additive_attention = AdditiveAttention(self.plm_config.hidden_size, additive_attn_hidden_dim)

    @property
    def embedding_size(self) -> int:
        return self.plm_config.hidden_size

    def forward(self, article_tokens: torch.Tensor) -> torch.Tensor:
        nonzero_inputs = article_tokens.bool()

        # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        token_embeddings = self.plm(article_tokens, attention_mask=nonzero_inputs.int()).last_hidden_state

        # Multi-head attention needs at least one position to be unmasked
        # or it returns NaNs, so unmask the first position even if it's padding
        # (Note: article headline tokens are padded on the right)
        mha_padding_mask = ~nonzero_inputs
        mha_padding_mask[:, 0] = False

        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, hidden_size]
        multihead_attn_output, _ = self.multihead_attention(
            token_embeddings, token_embeddings, token_embeddings, key_padding_mask=mha_padding_mask
        )

        # Additive attention can handle all positions being masked and will zero
        # the weights of the non-zero vectors coming from MHA in masked positions
        add_padding_mask = ~nonzero_inputs

        # [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size]
        article_vectors, _, _ = self.additive_attention(multihead_attn_output, padding_mask=add_padding_mask)

        return article_vectors
