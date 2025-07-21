import torch as th
from pytest import xfail
from safetensors.torch import load_file

from poprox_recommender.config import allow_data_test_failures
from poprox_recommender.model import ModelConfig
from poprox_recommender.model.nrms import NewsEncoder
from poprox_recommender.paths import model_file_path


def test_masked_padding_with_real_model():
    # Load the news encoder weights if model files are present
    try:
        checkpoint = load_file(model_file_path("nrms-mind/news_encoder.safetensors"))
        model_cfg = ModelConfig()
        news_encoder = NewsEncoder(
            "distilbert-base-uncased",
            model_cfg.num_attention_heads,
            model_cfg.additive_attn_hidden_dim,
        )
        news_encoder.load_state_dict(checkpoint)
    except FileNotFoundError as e:
        if allow_data_test_failures():
            xfail("data not pulled")
        else:
            raise e

    # Set up test data
    num_words = 25
    num_padding = 25

    words = th.randint(0, 30522, [1, num_words], dtype=th.int64)
    padding = th.zeros([1, num_padding], dtype=th.int64)
    news_input = th.concat([words, padding], dim=1)
    assert news_input.shape == th.Size([1, num_words + num_padding])

    padding_mask = ~news_input.bool()

    # DistilBERT expects 1s for unmasked positions and 0s for masked positions
    unmasked_word_embeddings = news_encoder.plm(news_input).last_hidden_state
    masked_word_embeddings = news_encoder.plm(news_input, attention_mask=~padding_mask.int()).last_hidden_state
    assert not th.equal(unmasked_word_embeddings, masked_word_embeddings)

    # Torch MHA expects True for masked positions and False for unmasked positions
    unmasked_mha_output, unmasked_mha_weights = news_encoder.multihead_attention(
        masked_word_embeddings, masked_word_embeddings, masked_word_embeddings
    )
    masked_mha_output, masked_mha_weights = news_encoder.multihead_attention(
        masked_word_embeddings, masked_word_embeddings, masked_word_embeddings, key_padding_mask=padding_mask
    )

    # Check that masking made a difference
    assert not th.equal(unmasked_mha_weights, masked_mha_weights)
    assert not th.equal(unmasked_mha_output, masked_mha_output)

    # Check that it made the difference we expect
    # Attention weights for all padding positions should be zero
    assert th.count_nonzero(masked_mha_weights[0, :, :num_words]) == th.numel(masked_mha_weights[0, :, num_words:])
    assert th.count_nonzero(masked_mha_weights[0, :, num_words:]) == 0

    # You'd think the output of multihead attention would be zero for padded positions,
    # but it isn't! You can check this by uncommenting the following lines:

    # contextualized_word_embeddings = masked_mha_output[0][:num_words]
    # padded_position_embeddings = masked_mha_output[0][num_words:]

    # assert th.count_nonzero(contextualized_word_embeddings) == th.numel(contextualized_word_embeddings)
    # assert th.count_nonzero(padded_position_embeddings) == 0

    # So instead we mask those positions at the input of the next layer
    unmasked_additive_sum, unmasked_additive_summands, unmasked_additive_weights = news_encoder.additive_attention(
        masked_mha_output
    )
    masked_additive_sum, masked_additive_summands, masked_additive_weights = news_encoder.additive_attention(
        masked_mha_output, padding_mask=padding_mask
    )

    # Check that masking made a difference
    assert not th.equal(unmasked_additive_weights, masked_additive_weights)
    assert not th.equal(unmasked_additive_summands, masked_additive_summands)

    # Check that it made the difference we expect
    # Attention weights for all padding positions should be zero
    assert th.count_nonzero(unmasked_additive_weights[0, :num_words, :]) == th.numel(
        unmasked_additive_weights[0, :num_words, :]
    )
    assert th.count_nonzero(masked_additive_weights[0, num_words:, :]) == 0

    # Weighted vectors for all padding positions should also be zero
    assert th.count_nonzero(unmasked_additive_summands[0, :num_words, :]) == th.numel(
        unmasked_additive_summands[0, :num_words, :]
    )
    assert th.count_nonzero(masked_additive_summands[0, num_words:, :]) == 0
