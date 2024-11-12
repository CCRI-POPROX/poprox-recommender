"""
Support for loading POPROX data for evaluation.
"""

# pyright: basic
from __future__ import annotations

import logging
from typing import Generator
from uuid import UUID

import pandas as pd

from poprox_concepts import Article, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.data.eval import EvalData
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10


class PoproxData(EvalData):
    clicks_df: pd.DataFrame
    articles_df: pd.DataFrame

    def __init__(self, archive: str = "POPROX"):
        articles_df, mentions_df, newsletters_df, clicks_df, clicked_articles_df, clicked_mentions_df = (
            load_poprox_frames(archive)
        )

        self.newsletters_df = newsletters_df

        # index data frames for quick lookup of users & articles
        self.mentions_df = mentions_df
        self.articles_df = articles_df.set_index("article_id", drop=False)
        if not self.articles_df.index.unique:
            logger.warning("article data has non-unique index")

        self.clicks_df = clicks_df
        self.clicked_mentions_df = clicked_mentions_df
        self.clicked_articles_df = clicked_articles_df.set_index("article_id", drop=False)
        if not self.clicked_articles_df.index.unique:
            logger.warning("clicked article data has non-unique index")

    @property
    def n_profiles(self) -> int:
        return len(self.newsletters_df["newsletter_id"].unique())

    @property
    def n_articles(self) -> int:
        return self.articles_df.shape[0]

    def profile_truth(self, newsletter_id: UUID) -> pd.DataFrame | None:
        # Create one row per clicked article with this newsletter_id
        # Returned dataframe must have an "item" column containing the clicked article ids
        # and the "item" column must be the index of the dataframe
        # There must also be a "rating" columns
        newsletter_clicks = self.clicks_df[self.clicks_df["newsletter_id"] == str(newsletter_id)]
        clicked_items = newsletter_clicks["article_id"].unique()
        return pd.DataFrame({"item": clicked_items, "rating": [1.0] * len(clicked_items)}).set_index("item")

    def iter_profiles(self) -> Generator[RecommendationRequest]:
        newsletter_ids = self.newsletters_df["newsletter_id"].unique()

        for newsletter_id in newsletter_ids:
            impressions_df = self.newsletters_df.loc[self.newsletters_df["newsletter_id"] == newsletter_id]
            # TODO: Change `account_id` to `profile_id` in the export
            profile_id = impressions_df.iloc[0]["account_id"]
            newsletter_created_at = impressions_df.iloc[0]["created_at"]

            # Filter clicks to those before the newsletter
            profile_clicks_df = self.clicks_df.loc[self.clicks_df["profile_id"] == profile_id]
            # TODO: Change `timestamp` to `created_at` in the export
            filtered_clicks_df = profile_clicks_df[profile_clicks_df["timestamp"] < newsletter_created_at]

            # Create Article and Click objects from dataframe rows
            clicks = []
            past_articles = []
            for article_row in filtered_clicks_df.itertuples():
                article = self.lookup_clicked_article(article_row.article_id)
                past_articles.append(article)

                clicks.append(
                    Click(
                        article_id=article_row.article_id,
                        newsletter_id=article_row.newsletter_id,
                        timestamp=article_row.timestamp,
                    )
                )

            # TODO: Fill in the onboarding topics
            profile = InterestProfile(profile_id=newsletter_id, click_history=clicks, onboarding_topics=[])

            # Filter candidate articles to those ingested on the same day as the newsletter (today's articles)
            candidate_articles = []
            newsletter_date = newsletter_created_at.date()

            for article_row in self.articles_df[
                self.articles_df["created_at"].apply(lambda c: c.date()) == newsletter_date
            ].itertuples():
                candidate_articles.append(self.lookup_candidate_article(article_row.article_id))

            yield RecommendationRequest(
                todays_articles=candidate_articles,
                past_articles=past_articles,
                interest_profile=profile,
                num_recs=TEST_REC_COUNT,
            )

    def lookup_candidate_article(self, article_id: UUID):
        article_row = self.articles_df.loc[str(article_id)]
        mention_rows = self.mentions_df[self.mentions_df["article_id"] == article_row.article_id]
        return self.convert_row_to_article(article_row, mention_rows)

    def lookup_clicked_article(self, article_id: UUID):
        article_row = self.clicked_articles_df.loc[str(article_id)]
        mention_rows = self.clicked_mentions_df[self.clicked_mentions_df["article_id"] == article_row.article_id]
        return self.convert_row_to_article(article_row, mention_rows)

    def convert_row_to_article(self, article_row, mention_rows):
        mentions = [
            Mention(
                mention_id=row.mention_id,
                article_id=row.article_id,
                source=row.source,
                relevance=row.relevance,
                entity=Entity(**row.entity),
            )
            for row in mention_rows.itertuples()
        ]

        return Article(
            article_id=article_row.article_id,
            headline=article_row.headline,
            subhead=article_row.subhead,
            published_at=article_row.published_at,
            mentions=mentions,
            source="AP",
            external_id="",
            raw_data=article_row.raw_data,
        )


def load_poprox_frames(archive: str = "POPROX"):
    data = project_root() / "data"
    logger.info("loading POPROX data from %s", archive)

    newsletters_df = pd.read_parquet(data / "POPROX" / "newsletters.parquet")

    articles_df = pd.read_parquet(data / "POPROX" / "articles.parquet")
    mentions_df = pd.read_parquet(data / "POPROX" / "mentions.parquet")

    clicks_df = pd.read_parquet(data / "POPROX" / "clicks.parquet")
    clicked_articles_df = pd.read_parquet(data / "POPROX" / "clicked" / "articles.parquet")
    clicked_mentions_df = pd.read_parquet(data / "POPROX" / "clicked" / "mentions.parquet")

    return articles_df, mentions_df, newsletters_df, clicks_df, clicked_articles_df, clicked_mentions_df
