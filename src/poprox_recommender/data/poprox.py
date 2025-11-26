"""
Support for loading POPROX data for evaluation.
"""

# pyright: basic
from __future__ import annotations

import itertools as it
import json
import logging
from typing import Generator
from uuid import UUID

import pandas as pd

from poprox_concepts.api.recommendations import RecommendationRequestV4
from poprox_concepts.domain import AccountInterest, Article, CandidateSet, Click, Entity, InterestProfile, Mention
from poprox_recommender.data.eval import EvalData
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10


class PoproxData(EvalData):
    name = "POPROX"

    def __init__(self, archive: str = "POPROX"):
        (
            articles_df,
            mentions_df,
            newsletters_df,
            clicks_df,
            clicked_articles_df,
            clicked_mentions_df,
            interests_df,
        ) = load_poprox_frames(archive)

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

        self.interests_df = interests_df

    @property
    def n_requests(self) -> int:
        return len(self.newsletters_df["newsletter_id"].unique())

    @property
    def n_articles(self) -> int:
        return self.articles_df.shape[0]

    def slate_truth(self, slate_id: UUID) -> pd.DataFrame | None:
        # Create one row per clicked article with this newsletter_id
        # Returned dataframe must have an "item_id" column containing the clicked article ids
        # and the "item_id" column must be the index of the dataframe
        # There must also be a "rating" columns
        newsletter_clicks = self.clicks_df[self.clicks_df["newsletter_id"] == str(slate_id)]
        clicked_items = newsletter_clicks["article_id"].unique()
        return pd.DataFrame({"item_id": clicked_items, "rating": [1.0] * len(clicked_items)}).set_index("item_id")

    def iter_slate_ids(self, *, limit: int | None = None) -> Generator[UUID]:
        # One profile/newsletter per account
        newsletters_df = self.newsletters_df.drop_duplicates(subset=["account_id"])

        newsletter_ids = newsletters_df["newsletter_id"].unique()
        if limit is not None:
            newsletter_ids = it.islice(newsletter_ids, limit)
        for id in newsletter_ids:
            if not isinstance(id, UUID):
                id = UUID(id)
            yield id

    def lookup_request(self, slate_id: UUID) -> RecommendationRequestV4:
        impressions_df = self.newsletters_df.loc[self.newsletters_df["newsletter_id"] == str(slate_id)]
        account_id = impressions_df.iloc[0]["account_id"]
        newsletter_created_at = impressions_df.iloc[0]["created_at"]

        # Filter clicks to those before the newsletter
        account_clicks_df = self.clicks_df.loc[self.clicks_df["account_id"] == str(account_id)]
        # TODO: Change `timestamp` to `created_at` in the export
        filtered_clicks_df = account_clicks_df[account_clicks_df["clicked_at"] < newsletter_created_at]

        # Create Article and Click objects from dataframe rows
        clicks = []
        past_articles = []
        for article_row in filtered_clicks_df.itertuples():
            article = self.lookup_clicked_article(article_row.article_id)
            if article:
                past_articles.append(article)

                clicks.append(
                    Click(
                        article_id=article_row.article_id,
                        newsletter_id=article_row.newsletter_id,
                        timestamp=article_row.clicked_at,
                    )
                )

        interests = self.interests_df.loc[self.interests_df["account_id"] == account_id]
        topics = []
        for interest in interests.itertuples():
            topics.append(
                AccountInterest(
                    account_id=account_id,
                    entity_id=interest.entity_id,
                    entity_name=interest.entity_name,
                    entity_type="topic",
                    preference=interest.preference,
                )
            )

        profile = InterestProfile(profile_id=slate_id, click_history=clicks, entity_interests=topics)

        # Filter candidate articles to those ingested on the same day as the newsletter (today's articles)
        candidate_articles = []
        newsletter_date = newsletter_created_at.date()

        for article_row in self.articles_df[
            self.articles_df["created_at"].apply(lambda c: c.date()) == newsletter_date
        ].itertuples():
            candidate_articles.append(self.lookup_candidate_article(article_row.article_id))

        return RecommendationRequestV4(
            candidates=CandidateSet(articles=candidate_articles),
            interacted=CandidateSet(articles=past_articles),
            interest_profile=profile,
            num_recs=TEST_REC_COUNT,
        )

    def lookup_candidate_article(self, article_id: UUID):
        article_row = self.articles_df.loc[str(article_id)]
        mention_rows = self.mentions_df[self.mentions_df["article_id"] == article_row.article_id]
        return self.convert_row_to_article(article_row, mention_rows)

    def lookup_clicked_article(self, article_id: UUID):
        try:
            article_row = self.clicked_articles_df.loc[str(article_id)]
            mention_rows = self.clicked_mentions_df[self.clicked_mentions_df["article_id"] == article_row.article_id]
            return self.convert_row_to_article(article_row, mention_rows)
        except Exception as _:
            print(f"Did not find the clicked article with id {str(article_id)}")
            return None

    def convert_row_to_article(self, article_row, mention_rows):
        mentions = [
            Mention(
                mention_id=row.mention_id,
                article_id=row.article_id,
                source=row.source,
                relevance=row.relevance,
                entity=Entity(**json.loads(row.entity)) if row.entity else None,
            )
            for row in mention_rows.itertuples()
        ]

        return Article(
            article_id=article_row.article_id,
            headline=article_row.headline,
            subhead=article_row.subhead,
            body=article_row.body,
            published_at=article_row.published_at,
            mentions=mentions,
            source="AP",
            external_id="",
            raw_data=json.loads(article_row.raw_data) if article_row.raw_data else None,
        )


def load_poprox_frames(archive: str = "POPROX"):
    data = project_root() / "data"
    logger.info("loading POPROX data from %s", archive)

    newsletters_df = pd.read_parquet(data / archive / "newsletters.parquet")

    articles_df = pd.read_parquet(data / archive / "articles.parquet")
    mentions_df = pd.read_parquet(data / archive / "article_mentions.parquet")

    clicks_df = pd.read_parquet(data / archive / "clicks.parquet")
    clicked_articles_df = pd.read_parquet(data / archive / "clicked" / "articles.parquet")
    clicked_mentions_df = pd.read_parquet(data / archive / "clicked" / "article_mentions.parquet")

    interests_df = pd.read_parquet(data / archive / "interests.parquet")

    return (
        articles_df,
        mentions_df,
        newsletters_df,
        clicks_df,
        clicked_articles_df,
        clicked_mentions_df,
        interests_df,
    )
