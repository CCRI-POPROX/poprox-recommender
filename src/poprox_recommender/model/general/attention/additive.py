import torch
from torch import nn


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(
                input_dim, hidden_dim
            ),  # in: (batch_size, seq_len, input_dim), out: (batch_size, seq_len, hidden_dim)
            nn.Tanh(),  # in: (batch_size, seq_len, hidden_dim), out: (batch_size, seq_len, hidden_dim)
            nn.Linear(
                hidden_dim, 1, bias=False
            ),  # in: (batch_size, seq_len, hidden_dim), out: (batch_size, seq_len, 1)
        )
        self.softmax = nn.Softmax(dim=-2)
        self.attention.apply(init_weights)

    def forward(
        self, input: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        softmax_input = self.attention(input)

        if padding_mask is not None:
            mask_values = torch.nan_to_num(-torch.inf * padding_mask)
            softmax_input = softmax_input + mask_values.unsqueeze(dim=2)

        attention_weights = self.softmax(softmax_input)
        attention_outputs = input * attention_weights
        weighted_sum = torch.sum(attention_outputs, dim=1)

        return weighted_sum, attention_outputs, attention_weights
