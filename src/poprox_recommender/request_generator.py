import random
from datetime import datetime, timedelta
from typing import List
from uuid import uuid4

from pydantic import ValidationError

from poprox_concepts import AccountInterest, Click, InterestProfile
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.data.mind import MindData


class RequestGenerator:
    """
    Class to generate recommendation request using click history, onboarding topics, and candidate articles from MIND
    """

    def __init__(self, mind_data: MindData):
        self.mind_data = mind_data
        self.candidate_articles = list()
        self.past_articles = list()
        self.added_topics = list()
        self.clicks = list()

    def set_num_recs(self, num_recs: int):
        self.num_recs = num_recs

    def add_clicks(self, num_clicks: int, num_days: int | None = None):
        all_articles = list(self.mind_data.news_df.index)

        if num_days:
            start_date = datetime.now() - timedelta(days=num_days - 1)
            timestamps = [start_date + timedelta(days=random.randint(0, num_days - 1)) for _ in range(num_clicks)]
            random.shuffle(timestamps)
        else:
            timestamps = [datetime.now()] * num_clicks
        # generate click history
        self.clicks = [
            Click(
                article_id=self.mind_data.news_uuid_for_id(random.choice(all_articles)),
                newsletter_id=uuid4(),
                timestamp=timestamps[i],
            )
            for i in range(num_clicks)
        ]

        self.past_articles = [self.mind_data.lookup_article(uuid=click.article_id) for click in self.clicks]

    def add_topics(self, topics: List[str]):
        self.added_topics = [
            AccountInterest(
                account_id=uuid4(),
                entity_id=uuid4(),
                entity_name=topic,
                preference=random.randint(1, 5),
                frequency=None,
            )
            for topic in topics
        ]

    def add_candidates(self, num_candidates):
        all_articles = list(self.mind_data.news_df.index)
        selected_candidates = random.sample(all_articles, num_candidates)

        self.candidate_articles = [self.mind_data.lookup_article(id=article_id) for article_id in selected_candidates]

    def get_request(self) -> RecommendationRequest:
        interest_profile = InterestProfile(
            profile_id=uuid4(),
            click_history=self.clicks,
            onboarding_topics=self.added_topics,
        )

        try:
            request = RecommendationRequest(
                past_articles=self.past_articles,
                todays_articles=self.candidate_articles,
                interest_profile=interest_profile,
                num_recs=self.num_recs,
            )
            return request
        except ValidationError as e:
            raise ValueError(f"Generated request is invalid: {e}")
