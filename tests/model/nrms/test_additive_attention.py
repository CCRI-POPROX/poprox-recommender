import torch as th

from poprox_recommender.model.general.attention.additive import AdditiveAttention


def test_padded_positions_have_zero_weights():
    attn = AdditiveAttention(128, 128)

    num_items = 25
    inputs = th.cat([th.rand((1, num_items, 128)), th.zeros((1, num_items, 128))], dim=1)
    assert inputs.shape == (1, 2 * num_items, 128)

    padding_mask = th.cat([th.zeros((1, 25)), th.ones((1, 25))], dim=1)

    masked_additive_sum, masked_additive_summands, masked_additive_weights = attn.forward(inputs, padding_mask)

    # Attention weights for all padding positions should be zero
    assert th.count_nonzero(masked_additive_weights[0, num_items:, :]) == 0

    # Weighted vectors for all padding positions should also be zero
    assert th.count_nonzero(masked_additive_summands[0, num_items:, :]) == 0

    # But the weighted sums should be non-zero
    assert th.count_nonzero(masked_additive_sum) == th.numel(masked_additive_sum)


def test_all_padding_gives_all_zero_weights():
    attn = AdditiveAttention(128, 128)

    inputs = th.rand((1, 50, 128))
    padding_mask = th.ones((1, 50))

    masked_additive, masked_additive_outputs, masked_additive_weights = attn.forward(inputs, padding_mask)

    # Regardless of the input vectors, when all positions are masked
    # then the weights, weighted vectors, and the final weighted sum
    # should all be zero throughout
    assert th.count_nonzero(masked_additive_weights) == 0
    assert th.count_nonzero(masked_additive_outputs) == 0
    assert th.count_nonzero(masked_additive) == 0
