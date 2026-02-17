from uuid import uuid4

import numpy as np

from poprox_concepts.domain import Article, CandidateSet
from poprox_recommender.components.filters.impression import ImpressionFilter


def test_impression_filter_removes_impressed_articles():
    """Test that impressed articles are filtered out from candidates."""
    # Create test articles
    article1 = Article(
        article_id=uuid4(),
        headline="Elon Musk",
        url="https://apnews.com/article/elon-musk-orbital-ai-data-centers-xai-spacex-92bc8ad95593bf3b5b801ddf36427194",
    )
    article2 = Article(
        article_id=uuid4(),
        headline="Iran US",
        url="https://apnews.com/article/iran-us-nuclear-talks-protests-araghchi-389531836ccaa4c126b5ee06c1d5b1f8",
    )
    article3 = Article(
        article_id=uuid4(),
        headline="Googles Results",
        url="https://apnews.com/article/google-alphabet-fourth-quarter-results-73922dd5d0c2398e1d4f23ddfccd0277",
    )
    article4 = Article(
        article_id=uuid4(),
        headline="AI OpenAI",
        url="https://apnews.com/article/openai-anthropic-chatgpt-claude-rivalry-c19e0cca22c37190cc4e0dc08e889ef0",
    )

    # Create candidate set with all 4 articles
    candidates = CandidateSet(articles=[article1, article2, article3, article4])

    # List of impressed article IDs (article1 and article2 already shown)
    impressed_article_ids = [article1.article_id, article2.article_id]

    # Apply filter
    filter_component = ImpressionFilter()
    filtered = filter_component(candidates, impressed_article_ids)

    # Assert only article3 and article4 remain
    assert len(filtered.articles) == 2
    assert article3 in filtered.articles
    assert article4 in filtered.articles
    assert article1 not in filtered.articles
    assert article2 not in filtered.articles


def test_impression_filter_empty_history():
    """Test that no filtering happens when impression history is empty."""
    # Create test articles
    article1 = Article(
        article_id=uuid4(),
        headline="Elon Musk",
        url="https://apnews.com/article/elon-musk-orbital-ai-data-centers-xai-spacex-92bc8ad95593bf3b5b801ddf36427194",
    )
    article2 = Article(
        article_id=uuid4(),
        headline="Iran US",
        url="https://apnews.com/article/iran-us-nuclear-talks-protests-araghchi-389531836ccaa4c126b5ee06c1d5b1f8",
    )
    # Create candidate set
    candidates = CandidateSet(articles=[article1, article2])

    # Empty list of impressed article IDs
    impressed_article_ids = []

    # Apply filter
    filter_component = ImpressionFilter()
    filtered = filter_component(candidates, impressed_article_ids)

    # Assert all articles remain
    assert len(filtered.articles) == 2
    assert article1 in filtered.articles
    assert article2 in filtered.articles


def test_impression_filter_all_articles_impressed():
    """Test behavior when all candidate articles have been impressed."""
    # Create test articles
    article1 = Article(
        article_id=uuid4(),
        headline="Elon Musk",
        url="https://apnews.com/article/elon-musk-orbital-ai-data-centers-xai-spacex-92bc8ad95593bf3b5b801ddf36427194",
    )
    article2 = Article(
        article_id=uuid4(),
        headline="Iran US",
        url="https://apnews.com/article/iran-us-nuclear-talks-protests-araghchi-389531836ccaa4c126b5ee06c1d5b1f8",
    )
    # Create candidate set
    candidates = CandidateSet(articles=[article1, article2])

    # All articles have been impressed
    impressed_article_ids = [article1.article_id, article2.article_id]

    # Apply filter
    filter_component = ImpressionFilter()
    filtered = filter_component(candidates, impressed_article_ids)

    # Assert no articles remain (empty candidate set)
    assert len(filtered.articles) == 0
    # Scores should be None or empty
    assert filtered.scores is None or len(filtered.scores) == 0


def test_impression_filter_no_overlap():
    """Test when impressed articles don't overlap with candidates."""
    # Create test articles
    article1 = Article(
        article_id=uuid4(),
        headline="Elon Musk",
        url="https://apnews.com/article/elon-musk-orbital-ai-data-centers-xai-spacex-92bc8ad95593bf3b5b801ddf36427194",
    )
    article2 = Article(
        article_id=uuid4(),
        headline="Iran US",
        url="https://apnews.com/article/iran-us-nuclear-talks-protests-araghchi-389531836ccaa4c126b5ee06c1d5b1f8",
    )

    # Create different impressed article IDs (not matching any candidates)
    impressed_id1 = uuid4()
    impressed_id2 = uuid4()

    # Create candidate set
    candidates = CandidateSet(articles=[article1, article2])

    # List of impressed article IDs that don't match candidates
    impressed_article_ids = [impressed_id1, impressed_id2]

    # Apply filter
    filter_component = ImpressionFilter()
    filtered = filter_component(candidates, impressed_article_ids)

    # Assert all articles remain (no overlap)
    assert len(filtered.articles) == 2
    assert article1 in filtered.articles
    assert article2 in filtered.articles
