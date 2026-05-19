from uuid import UUID

import pandas as pd
import pytest

from poprox_concepts.domain import Article, ImpressedSection, Impression
from poprox_recommender.evaluation.metrics.distinct_sections_clicked import distinct_sections_clicked


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


def test_no_clicks():
    truth = make_truth([], ALL_IDS)
    assert distinct_sections_clicked(SECTIONS, truth) == 0


def test_click_in_one_section():
    truth = make_truth([ALL_IDS[0]], ALL_IDS)
    assert distinct_sections_clicked(SECTIONS, truth) == 1


def test_clicks_in_two_sections():
    truth = make_truth([ALL_IDS[0], ALL_IDS[3]], ALL_IDS)
    assert distinct_sections_clicked(SECTIONS, truth) == 2


def test_clicks_in_three_sections():
    truth = make_truth([ALL_IDS[0], ALL_IDS[3], ALL_IDS[6]], ALL_IDS)
    assert distinct_sections_clicked(SECTIONS, truth) == 3


def test_all_sections_clicked():
    truth = make_truth([ALL_IDS[0], ALL_IDS[3], ALL_IDS[6], ALL_IDS[9], ALL_IDS[12]], ALL_IDS)
    assert distinct_sections_clicked(SECTIONS, truth) == 5


def test_multiple_clicks_same_section_counts_once():
    truth = make_truth([ALL_IDS[0], ALL_IDS[1], ALL_IDS[2]], ALL_IDS)
    assert distinct_sections_clicked(SECTIONS, truth) == 1


def test_click_on_article_not_in_sections():
    outside_id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
    truth = pd.DataFrame({"rating": [1]}, index=[outside_id])
    assert distinct_sections_clicked(SECTIONS, truth) == 0


def test_empty_sections():
    truth = make_truth([ALL_IDS[0]], ALL_IDS)
    assert distinct_sections_clicked([], truth) == 0
