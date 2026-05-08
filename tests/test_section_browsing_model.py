from uuid import uuid4

import numpy as np
import pytest

from poprox_concepts.domain import Article, ImpressedSection, Impression
from poprox_recommender.evaluation.measure.section_browsing_model import section_browsing_weights


def make_section(n_articles: int) -> ImpressedSection:
    impressions = [Impression(article=Article(article_id=uuid4(), headline=f"Article {i}")) for i in range(n_articles)]
    return ImpressedSection(impressions=impressions)


def test_weights_shape():
    sections = [make_section(3) for _ in range(5)]
    weights = section_browsing_weights(sections)
    assert len(weights) == 15


def test_weights():
    """
    Verify weights match:
    alpha=0.10, sigma=0.20, gamma=0.70
    """
    sections = [make_section(3) for _ in range(5)]
    weights = section_browsing_weights(sections, alpha=0.10, sigma=0.20)

    expected = np.array(
        [
            0.700,
            0.490,
            0.343,  # Section 1
            0.52269,
            0.365883,
            0.256118,  # Section 2
            0.390293,
            0.273205,
            0.191243,  # Section 3
            0.291432,
            0.204002,
            0.142801,  # Section 4
            0.217612,
            0.152328,
            0.10663,  # Section 5
        ]
    )

    np.testing.assert_allclose(weights, expected, rtol=1e-2)


def test_weights_decay_within_section():
    sections = [make_section(3)]
    weights = section_browsing_weights(sections, alpha=0.10, sigma=0.20)
    assert weights[0] > weights[1] > weights[2]


def test_weights_decay_across_sections():
    sections = [make_section(3) for _ in range(3)]
    weights = section_browsing_weights(sections, alpha=0.10, sigma=0.20)
    # first item of each section
    assert weights[0] > weights[3] > weights[6]


def test_weights_all_positive():
    sections = [make_section(3) for _ in range(5)]
    weights = section_browsing_weights(sections, alpha=0.15, sigma=0.30)
    assert np.all(weights > 0)


def test_weights_single_section():
    """Single section degrades to geometric decay by gamma."""
    sections = [make_section(3)]
    gamma = 0.70
    weights = section_browsing_weights(sections, alpha=0.10, sigma=0.20)
    expected = np.array([gamma, gamma**2, gamma**3])
    np.testing.assert_allclose(weights, expected, rtol=1e-10)
