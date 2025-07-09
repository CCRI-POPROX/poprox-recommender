import json
from uuid import uuid4

from poprox_concepts import AccountInterest, InterestProfile
from poprox_recommender.paths import project_root

with open(project_root() / "tests/request_data/onboarding.json", "r") as req_file:
    base_request_data = json.load(req_file)


topics = base_request_data["interest_profile"]["onboarding_topics"]

single_topic_personas = {}
for persona in topics:
    topic_profile = InterestProfile(
        profile_id=uuid4(),
        click_history=[],
        click_topic_counts=None,
        click_locality_counts=None,
        article_feedbacks={},
        onboarding_topics=[],
    )

    for topic in topics:
        preference = 5 if topic["entity_name"] == persona["entity_name"] else 1
        topic_profile.onboarding_topics.append(
            AccountInterest(entity_id=topic["entity_id"], entity_name=topic["entity_name"], preference=preference)
        )
    single_topic_personas[persona["entity_name"]] = topic_profile
