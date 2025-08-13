"""
Support for loading POPROX data for evaluation.
"""

# pyright: basic
from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from itertools import chain, product
from typing import Generator, Tuple
from uuid import UUID

import numpy as np
import pandas as pd

from poprox_concepts import AccountInterest, Article, Click, Entity, InterestProfile, Mention
from poprox_concepts.api.recommendations import RecommendationRequestV2
from poprox_concepts.domain import CandidateSet
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
            experiment_df,
            assignments_df,
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

        calibration_group_ids = experiment_df.loc[(experiment_df["group_name"] == "calibration_treatment")][
            "group_id"
        ].unique()

        # Step 2: Get profile_ids in that group
        calibration_profile_ids = assignments_df.loc[assignments_df["group_id"].isin(calibration_group_ids)][
            "profile_id"
        ].unique()

        logger.info("Downsampling to %d profiles in calibration group...", len(calibration_profile_ids))

        # Step 3: Subset all relevant DataFrames
        self.newsletters_df = newsletters_df[newsletters_df["profile_id"].isin(calibration_profile_ids)]
        # self.clicks_df = clicks_df[clicks_df["profile_id"].isin(calibration_profile_ids)]
        # self.clicked_mentions_df = clicked_mentions_df[
        #     clicked_mentions_df["article_id"].isin(self.clicks_df["article_id"])
        # ]
        # self.clicked_articles_df = clicked_articles_df[
        #     clicked_articles_df["article_id"].isin(self.clicks_df["article_id"])
        # ]
        # self.interests_df = interests_df[interests_df["account_id"].isin(calibration_profile_ids)]

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
        # Returned dataframe must have an "item_id" column containing the clicked article ids
        # and the "item_id" column must be the index of the dataframe
        # There must also be a "rating" columns
        newsletter_clicks = self.clicks_df[self.clicks_df["newsletter_id"] == str(newsletter_id)]
        clicked_items = newsletter_clicks["article_id"].unique()
        return pd.DataFrame({"item_id": clicked_items, "rating": [1.0] * len(clicked_items)}).set_index("item_id")

    def iter_hyperparameters_theta(
        self,
        topic_thetas: Tuple[float, float],
        topic_theta_incr: float,
        locality_thetas: Tuple[float, float],
        locality_theta_incr: float,
        random_sample: int | None = None,
    ) -> Generator[Tuple[RecommendationRequestV2, Tuple[float, float]], None, None]:
        """
        A wrapper around iter_profiles that extends its results with combinations of topic and locality thetas.

        Args:
            topic_thetas (Tuple[float, float]): Start and end values (inclusive) for topic theta range.
            topic_theta_incr (float): Increment for topic theta range.
            locality_thetas (Tuple[float, float]): Start and end values (inclusive) for locality theta range.
            locality_theta_incr (float): Increment for locality theta range.
            random_sample (int | None): If provided, randomly sample this many points from
                                        the Cartesian product of thetas.

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

    def iter_hyperparameters_theshold(
        self,
        similarity_thresholds: Tuple[float, float],
        similarity_threshold_incr: float,
    ) -> Generator[Tuple[RecommendationRequestV2, float], None, None]:
        """
        A wrapper around iter_profiles that extends its results with combinations of similarity thresholds.

        Args:
            similarity_thresholds (Tuple[float, float]): Start and end values (inclusive) for threshold range.
            similarity_threshold_incr (float): Increment for similarity threshold range.

        Yields:
            Tuple[RecommendationRequest, Tuple[float, float]]: A recommendation request paired with theta values.
        """
        # Generate lists of threshold values
        threshold_combinations = np.arange(
            similarity_thresholds[0], similarity_thresholds[1] + 0.01, similarity_threshold_incr
        )

        logger.info(
            f"Generated {len(threshold_combinations)} similarity threshold combinations: {threshold_combinations}"  # noqa: E501
        )

        self.num_hyperparameters = len(threshold_combinations)

        # Extend iter_profiles to iterate over each profile with each theta combination
        profile_iter = self.iter_profiles()
        extended_iter = chain.from_iterable(
            ((profile, threshold) for threshold in threshold_combinations) for profile in profile_iter
        )

        for profile_with_threshold in extended_iter:
            yield profile_with_threshold

    def iter_profiles(
        self,
    ) -> Generator[RecommendationRequestV2, None, None]:
        newsletter_ids = self.newsletters_df["newsletter_id"].unique()

        for newsletter_id in newsletter_ids:
            impressions_df = self.newsletters_df.loc[self.newsletters_df["newsletter_id"] == newsletter_id]
            profile_id = impressions_df.iloc[0]["profile_id"]
            newsletter_created_at = impressions_df.iloc[0]["created_at"]

            # Filter clicks to those before the newsletter
            profile_clicks_df = self.clicks_df.loc[self.clicks_df["profile_id"] == profile_id]
            logger.info(profile_clicks_df.columns)
            filtered_clicks_df = profile_clicks_df[profile_clicks_df["clicked_at"] < newsletter_created_at]
            if len(filtered_clicks_df) == 0:
                logger.warning(f"No clicks for profile {profile_id} before newsletter {newsletter_id}")
                continue

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

            interests = self.interests_df.loc[self.interests_df["account_id"] == profile_id]
            topics = []
            for interest in interests.itertuples():
                topics.append(
                    AccountInterest(
                        account_id=profile_id,
                        entity_id=interest.entity_id,
                        entity_name=interest.entity_name,
                        preference=interest.preference,
                        # frequency=interest.frequency if not math.isnan(interest.frequency) else -1,
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

            yield (
                RecommendationRequestV2(
                    candidates=CandidateSet(articles=candidate_articles),
                    interacted=CandidateSet(articles=past_articles),
                    interest_profile=profile,
                    num_recs=TEST_REC_COUNT,  # Check to make sure None inputs are ok in the diversifier
                ),
                newsletter_id,
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
            print(f"Exception {_}")
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


def load_poprox_frames(archive: str = "POPROX", start_date: datetime | None = None, end_date: datetime | None = None):
    data = project_root() / "data"
    logger.info("loading POPROX data from %s", archive)

    newsletters_df = pd.read_parquet(
        data / "POPROX" / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1" / "newsletters_20250717-194301.parquet"
    )
    newsletters_df["created_at_date"] = pd.to_datetime(newsletters_df["created_at"])

    if start_date:
        logger.info("loading newsleters on or after %s", start_date)
        newsletters_df = newsletters_df[newsletters_df["created_at_date"] >= start_date]
    if end_date:
        logger.info("loading newsleters before %s", end_date)
        newsletters_df = newsletters_df[newsletters_df["created_at_date"] < end_date]

    articles_df = pd.read_parquet(
        data
        / "POPROX"
        / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1"
        / "candidate"
        / "articles_20250717-194309.parquet"
    )
    mentions_df = pd.read_parquet(
        data
        / "POPROX"
        / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1"
        / "candidate"
        / "article_mentions_20250717-194310.parquet"
    )

    clicks_df = pd.read_parquet(
        data / "POPROX" / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1" / "clicks_20250717-194312.parquet"
    )
    clicked_articles_df = pd.read_parquet(
        data
        / "POPROX"
        / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1"
        / "clicked"
        / "articles_20250717-194314.parquet"
    )
    clicked_mentions_df = pd.read_parquet(
        data
        / "POPROX"
        / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1"
        / "clicked"
        / "article_mentions_20250717-194315.parquet"
    )

    interests_df = pd.read_parquet(
        data / "POPROX" / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1" / "interests_20250717-194159.parquet"
    )

    experiment_df = pd.read_parquet(
        data / "POPROX" / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1" / "experiment_20250717-194157.parquet"
    )
    assignments_df = pd.read_parquet(
        data / "POPROX" / "experiment-0fc61eca-fca7-4914-9ea0-bc29ed2e0ad1" / "assignments_20250717-194157.parquet"
    )

    return (
        articles_df,
        mentions_df,
        newsletters_df,
        clicks_df,
        clicked_articles_df,
        clicked_mentions_df,
        interests_df,
        experiment_df,
        assignments_df,
    )
