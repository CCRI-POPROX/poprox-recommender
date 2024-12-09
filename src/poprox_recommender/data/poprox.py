"""
Support for loading POPROX data for evaluation.
"""

# pyright: basic
from __future__ import annotations

import logging
import random
from datetime import datetime
from itertools import chain, product
from typing import Generator, Tuple
from uuid import UUID

import numpy as np
import pandas as pd

from poprox_concepts import AccountInterest, Article, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.data.eval import EvalData
from poprox_recommender.paths import project_root

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10


class PoproxData(EvalData):
    def __init__(
        self,
        archive: str = "POPROX",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ):
        (
            articles_df,
            mentions_df,
            newsletters_df,
            clicks_df,
            clicked_articles_df,
            clicked_mentions_df,
            interests_df,
        ) = load_poprox_frames(archive, start_date, end_date)

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

        # Default to 1 for when iter_hyperparameters is never called
        self.num_hyperparameters = 1

    @property
    def n_profiles(self) -> int:
        return len(self.newsletters_df["newsletter_id"].unique())

    @property
    def n_hyperparameters(self) -> int:
        return self.num_hyperparameters

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

    def iter_hyperparameters(
        self,
        topic_thetas: Tuple[float, float],
        topic_theta_incr: float,
        locality_thetas: Tuple[float, float],
        locality_theta_incr: float,
        random_sample: int | None = None,
    ) -> Generator[Tuple[RecommendationRequest, Tuple[float, float]], None, None]:
        """
        A wrapper around iter_profiles that extends its results with combinations of topic and locality thetas.

        Args:
            topic_thetas (Tuple[float, float]): Start and end values (inclusive) for topic theta range.
            topic_theta_incr (float): Increment for topic theta range.
            locality_thetas (Tuple[float, float]): Start and end values (inclusive) for locality theta range.
            locality_theta_incr (float): Increment for locality theta range.
            random_sample (int | None): If provided, randomly sample this many points from the Cartesian product of thetas.

        Yields:
            Tuple[RecommendationRequest, Tuple[float, float]]: A recommendation request paired with theta values.
        """
        # Generate lists of theta values
        topic_range = np.arange(topic_thetas[0], topic_thetas[1] + 0.01, topic_theta_incr)
        locality_range = np.arange(locality_thetas[0], locality_thetas[1] + 0.01, locality_theta_incr)

        theta_combinations = list(product(topic_range, locality_range))

        logger.info(
            f"Generated cross product of topic_theta:{topic_thetas} X locality_theta:{locality_thetas} resulting in {len(theta_combinations)} combinations"  # noqa: E501
        )

        # Select a random sample of the full combinations
        # @mdekstrand 60 is sufficient for 95% confidence
        if random_sample is not None and random_sample < len(theta_combinations):
            theta_combinations = random.sample(theta_combinations, random_sample)
            logger.info(
                f"Down sampling to {random_sample} random combinations"  # noqa: E501
            )

        self.num_hyperparameters = len(theta_combinations)

        # Extend iter_profiles to iterate over each profile with each theta combination
        profile_iter = self.iter_profiles()
        extended_iter = chain.from_iterable(
            ((profile, theta) for theta in theta_combinations) for profile in profile_iter
        )

        for profile_with_theta in extended_iter:
            yield profile_with_theta

    def iter_profiles(
        self,
    ) -> Generator[RecommendationRequest, None, None]:
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

            interests = self.interests_df.loc[self.interests_df["account_id"] == profile_id]
            topics = []
            for interest in interests.itertuples():
                topics.append(
                    AccountInterest(
                        account_id=profile_id,
                        entity_id=interest.entity_id,
                        entity_name=interest.entity_name,
                        preference=interest.preference,
                        frequency=interest.frequency,
                    )
                )

            profile = InterestProfile(profile_id=newsletter_id, click_history=clicks, onboarding_topics=topics)

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
                num_recs=TEST_REC_COUNT,  # Check to make sure None inputs are ok in the diversifier
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


def load_poprox_frames(archive: str = "POPROX", start_date: datetime | None = None, end_date: datetime | None = None):
    data = project_root() / "data"
    logger.info("loading POPROX data from %s", archive)

    newsletters_df = pd.read_parquet(data / "POPROX" / "newsletters.parquet")
    newsletters_df["created_at_date"] = pd.to_datetime(newsletters_df["created_at"])

    if start_date:
        logger.info("loading newsleters on or after %s", start_date)
        newsletters_df = newsletters_df[newsletters_df["created_at_date"] >= start_date]
    if end_date:
        logger.info("loading newsleters before %s", end_date)
        newsletters_df = newsletters_df[newsletters_df["created_at_date"] < end_date]

    articles_df = pd.read_parquet(data / "POPROX" / "articles.parquet")
    mentions_df = pd.read_parquet(data / "POPROX" / "mentions.parquet")

    clicks_df = pd.read_parquet(data / "POPROX" / "clicks.parquet")
    clicked_articles_df = pd.read_parquet(data / "POPROX" / "clicked" / "articles.parquet")
    clicked_mentions_df = pd.read_parquet(data / "POPROX" / "clicked" / "mentions.parquet")

    interests_df = pd.read_parquet(data / "POPROX" / "interests.parquet")

    return (
        articles_df,
        mentions_df,
        newsletters_df,
        clicks_df,
        clicked_articles_df,
        clicked_mentions_df,
        interests_df,
    )
