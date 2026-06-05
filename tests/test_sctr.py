from uuid import UUID

import pandas as pd
import pytest

from poprox_concepts.domain import Article, ImpressedSection, Impression
from poprox_recommender.evaluation.metrics.sctr import section_click_through_rate


def make_article(id_str: str) -> Article:
    return Article(article_id=UUID(id_str), headline="")


def make_section(articles: list[Article]) -> ImpressedSection:
    return ImpressedSection(impressions=[Impression(article=a) for a in articles])


def make_truth(clicked_ids: list[str], all_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"rating": [1 if aid in clicked_ids else 0 for aid in all_ids]},
        index=all_ids,
    )


ARTICLES = [make_article(f"00000000-0000-0000-0000-0000000000{str(i).zfill(2)}") for i in range(1, 16)]
ALL_IDS = [str(a.article_id) for a in ARTICLES]

# 5 sections of 3 articles each
SECTIONS = [
    make_section(ARTICLES[0:3]),  # section 0: articles 0, 1, 2
    make_section(ARTICLES[3:6]),  # section 1: articles 3, 4, 5
    make_section(ARTICLES[6:9]),  # section 2: articles 6, 7, 8
    make_section(ARTICLES[9:12]),  # section 3: articles 9, 10, 11
    make_section(ARTICLES[12:15]),  # section 4: articles 12, 13, 14
]


def test_no_clicks_returns_zero():
    truth = make_truth([], ALL_IDS)
    assert section_click_through_rate(SECTIONS, truth) == pytest.approx(0.0)


def test_all_articles_clicked_returns_one():
    truth = make_truth(ALL_IDS, ALL_IDS)
    assert section_click_through_rate(SECTIONS, truth) == pytest.approx(1.0)


def test_one_section_fully_clicked():
    # section 0: 3/3 clicked, rest 0/3 → mean([1.0, 0.0, 0.0, 0.0, 0.0]) = 0.2
    truth = make_truth(ALL_IDS[0:3], ALL_IDS)
    assert section_click_through_rate(SECTIONS, truth) == pytest.approx(0.2)


def test_one_article_per_section_clicked():
    # each section has 1/3 clicked → mean([1/3]*5) = 1/3
    clicked = [ALL_IDS[0], ALL_IDS[3], ALL_IDS[6], ALL_IDS[9], ALL_IDS[12]]
    truth = make_truth(clicked, ALL_IDS)
    assert section_click_through_rate(SECTIONS, truth) == pytest.approx(1 / 3)


def test_mixed_section_ctrs():
    # section 0: 3/3=1.0, section 1: 0/3=0.0, section 2: 1/3, section 3: 0/3=0.0, section 4: 2/3
    # mean = (1.0 + 0.0 + 1/3 + 0.0 + 2/3) / 5 = 2.0 / 5 = 0.4
    clicked = ALL_IDS[0:3] + [ALL_IDS[6]] + ALL_IDS[12:14]
    truth = make_truth(clicked, ALL_IDS)
    assert section_click_through_rate(SECTIONS, truth) == pytest.approx(0.4)


def test_click_outside_sections_ignored():
    outside_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    truth = pd.DataFrame({"rating": [1]}, index=[outside_id])
    assert section_click_through_rate(SECTIONS, truth) == pytest.approx(0.0)


def test_empty_sections_returns_zero():
    truth = make_truth([ALL_IDS[0]], ALL_IDS)
    assert section_click_through_rate([], truth) == pytest.approx(0.0)


def test_section_with_no_impressions_is_skipped():
    sections = [make_section(ARTICLES[0:3]), ImpressedSection(impressions=[])]
    truth = make_truth(ALL_IDS[0:3], ALL_IDS)
    # only the non-empty section contributes: [1.0] → mean = 1.0
    assert section_click_through_rate(sections, truth) == pytest.approx(1.0)
