import random
from typing import List
from uuid import uuid4

from pydantic import ValidationError

from poprox_concepts import AccountInterest, Click, InterestProfile
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.data.mind import MindData


class RequestGenerator:
    def __init__(self, mind_data: MindData, num_recs: int = 10):
        self.mind_data = mind_data
        self.num_recs = num_recs
        self.candidate_articles = []
        self.past_articles = []
        self.added_topics = []
        self.clicks = []

    def set_num_recs(self, num_recs: int):
        self.num_recs = num_recs

    def add_clicks(self, num_clicks: int):
        all_articles = list(self.mind_data.news_df.index)
        self.clicks = [
            Click(article_id=self.mind_data.news_uuid_for_id(random.choice(all_articles))) for _ in range(num_clicks)
        ]

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

    def add_candidates(self, num_candidates: int):
        all_articles = list(self.mind_data.news_df.index)
        selected_candidates = random.sample(all_articles, num_candidates)

        self.past_articles = [self.mind_data.lookup_article(uuid=click.article_id) for click in self.clicks]
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


mind_data = MindData()
request_generator = RequestGenerator(mind_data)

user_uuid = random.choice(list(mind_data.behavior_id_map.keys()))

request_generator.set_num_recs(2)
request_generator.add_clicks(2)
request_generator.add_topics([])
request_generator.add_candidates(num_candidates=3)

request = request_generator.get_request()

print(request)
