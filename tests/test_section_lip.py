import logging
from datetime import datetime
from uuid import UUID

import numpy as np
import pytest

from poprox_concepts.domain import Article, ImpressedSection, Impression
from poprox_recommender.evaluation.section_metrics.section_lip import _lip_for_section, section_wise_lip

logger = logging.getLogger(__name__)


def make_article(id_str):
    return Article(
        article_id=UUID(id_str),
        headline="",
        subhead=None,
        body=None,
        url=None,
        preview_image_id=None,
        mentions=[],
        source=None,
        external_id=None,
        raw_data=None,
        images=[],
        published_at=datetime.now(),
        created_at=None,
    )


def make_section(articles: list[Article]) -> ImpressedSection:
    """Wrap a list of articles into an ImpressedSection with one Impression each."""
    impressions = [Impression(article=a) for a in articles]
    return ImpressedSection(impressions=impressions)


@pytest.fixture
def all_articles():
    return [make_article(f"00000000-0000-0000-0000-0000000000{str(i).zfill(2)}") for i in range(1, 16)]


@pytest.fixture
def global_ids(all_articles):
    """Global fusion ranking: articles in order, rank 0 = best."""
    return [str(a.article_id) for a in all_articles]


# --- _lip_for_section tests ---


def test_lip_section_top3_taken_directly(all_articles, global_ids):
    # articles 0,1,2 are the top-3 globally → no promotion → LIP = 0
    section = make_section(all_articles[:3])
    rank_lookup = {aid: rank for rank, aid in enumerate(global_ids)}
    score = _lip_for_section(rank_lookup, section, k=3)
    assert score == 0.0


def test_lip_section_article_pulled_from_rank5(all_articles, global_ids):
    # section contains global ranks 0, 1, 5 → least promoted is at rank 5
    # LIP = 5 - 3 = 2
    section = make_section([all_articles[0], all_articles[1], all_articles[5]])
    rank_lookup = {aid: rank for rank, aid in enumerate(global_ids)}
    score = _lip_for_section(rank_lookup, section, k=3)
    assert score == 2.0


def test_lip_section_article_not_in_global_list(all_articles, global_ids):
    # one article in the section is not in global_ids → skipped, LIP based on others
    # all_articles[0] is at global rank 0 → no promotion beyond k=3 → LIP = 0
    extra = make_article("ffffffff-ffff-ffff-ffff-ffffffffffff")
    section = make_section([all_articles[0], extra])
    rank_lookup = {aid: rank for rank, aid in enumerate(global_ids)}
    score = _lip_for_section(rank_lookup, section, k=3)
    assert score == 0.0


def test_lip_section_empty(all_articles, global_ids):
    section = make_section([])
    rank_lookup = {aid: rank for rank, aid in enumerate(global_ids)}
    score = _lip_for_section(rank_lookup, section, k=3)
    assert np.isnan(score)


# --- section_wise_lip tests ---


def test_section_wise_lip_no_promotion_single_section(all_articles, global_ids):
    # section takes the global top-3 exactly → no promotion → LIP = 0
    section = make_section(all_articles[0:3])
    score = section_wise_lip(global_ids, [section], k=3)
    assert score == 0.0


def test_section_wise_lip_consecutive_sections(all_articles, global_ids):
    # section1: global ranks 0,1,2 → LIP = 0
    # section2: global ranks 3,4,5 → rank 5 > k=3 → LIP = 5 - 3 = 2
    # average = (0 + 2) / 2 = 1.0
    section1 = make_section(all_articles[0:3])
    section2 = make_section(all_articles[3:6])
    score = section_wise_lip(global_ids, [section1, section2], k=3)
    assert score == pytest.approx(1.0)


def test_section_wise_lip_one_promoted_section(all_articles, global_ids):
    # section1: ranks 0,1,2 → LIP = 0
    # section2: ranks 0,1,10 → LIP = 10 - 3 = 7
    # average = (0 + 7) / 2 = 3.5
    section1 = make_section(all_articles[0:3])
    section2 = make_section([all_articles[0], all_articles[1], all_articles[10]])
    score = section_wise_lip(global_ids, [section1, section2], k=3)
    assert score == pytest.approx(3.5)


def test_section_wise_lip_configurable_k(all_articles, global_ids):
    # with k=2, only first 2 articles per section are evaluated
    # section uses ranks 0, 1, 10 but k=2 → only ranks 0,1 checked → LIP = 0
    section = make_section([all_articles[0], all_articles[1], all_articles[10]])
    score = section_wise_lip(global_ids, [section], k=2)
    assert score == 0.0


def test_section_wise_lip_empty_global_list(all_articles):
    section = make_section(all_articles[0:3])
    score = section_wise_lip([], [section], k=3)
    assert np.isnan(score)


def test_section_wise_lip_empty_sections(global_ids):
    score = section_wise_lip(global_ids, [], k=3)
    assert np.isnan(score)


def test_section_wise_lip_averages_across_all_sections(all_articles, global_ids):
    # section1: ranks 0,1,2 → LIP = 0
    # section2: ranks 0,1,5 → LIP = 2
    # section3: ranks 0,1,9 → LIP = 6
    # average = (0 + 2 + 6) / 3 = 2.667
    section1 = make_section(all_articles[0:3])
    section2 = make_section([all_articles[0], all_articles[1], all_articles[5]])
    section3 = make_section([all_articles[0], all_articles[1], all_articles[9]])
    score = section_wise_lip(global_ids, [section1, section2, section3], k=3)
    assert score == pytest.approx(8 / 3)
