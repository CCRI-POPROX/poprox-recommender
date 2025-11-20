"""
With exported dataset, this script can help generate test request body for recommender pipieline testing
The data export dataset needs to be placed in side poprox-recommender/data/POPROX folder, with timestamp removed

Usage:
    generate_test_request.py [--account_id ID] [--min_click_num NUM] [--with_topics True] [--output_file OUTPUT]

Options:
    --account_id ID         Specific user account id to process request data
    --min_click_num NUM     Get a random user with at least min_click_num clicks
    --with_topics True      If True, indicating filter out users with a list of onboarding topics
    --output_file OUTPUT    Path to the output file
"""

import json
import random

from docopt import docopt

from poprox_concepts.api.recommendations import RecommendationRequestV4
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.paths import project_root


def iter_requests(eval_data, *, limit: int | None = None):
    for slate_id in eval_data.iter_slate_ids(limit=limit):
        yield eval_data.lookup_request(id=slate_id)


def get_single_request() -> str:
    options = docopt(__doc__)
    eval_data = PoproxData()
    requests = list(iter_requests(eval_data))
    excluded_fields = {"__all__": {"raw_data": True, "images": {"__all__": {"raw_data"}}}}

    account_id = options["--account_id"]
    min_click = options["--min_click_num"]
    with_topics = options["--with_topics"]
    output = options["--output_file"]

    request_body = ""
    if account_id:
        for req in requests:
            if req.interest_profile.profile_id == account_id:
                request_body = RecommendationRequestV4.model_dump_json(
                    req,
                    exclude={"candidates": excluded_fields, "interacted": excluded_fields, "protocol_version": True},
                )
                break
    elif min_click:
        if with_topics:
            for req in requests:
                if len(req.interest_profile.click_history) >= int(min_click) and req.interest_profile.interests_by_type(
                    "topic"
                ):
                    request_body = RecommendationRequestV4.model_dump_json(
                        req,
                        exclude={
                            "candidates": excluded_fields,
                            "interacted": excluded_fields,
                            "protocol_version": True,
                        },
                    )
                    break
        else:
            for req in requests:
                if len(req.interest_profile.click_history) >= int(min_click):
                    request_body = RecommendationRequestV4.model_dump_json(
                        req,
                        exclude={
                            "candidates": excluded_fields,
                            "interacted": excluded_fields,
                            "protocol_version": True,
                        },
                    )
                    break
    elif with_topics:
        for req in requests:
            if req.interest_profile.interests_by_type("topic"):
                request_body = RecommendationRequestV4.model_dump_json(
                    req,
                    exclude={"candidates": excluded_fields, "interacted": excluded_fields, "protocol_version": True},
                )
                break
    else:
        random_index = random.randint(0, len(requests) - 1)
        request_body = RecommendationRequestV4.model_dump_json(
            requests[random_index],
            exclude={"candidates": excluded_fields, "interacted": excluded_fields, "protocol_version": True},
        )

    if request_body:
        print("Found qualified test request!")

    if output:
        with open(output, "w") as file:
            file.write(request_body)
    else:
        request_data_path = project_root() / "tests" / "request_data" / "request_body_1.json"
        with open(request_data_path, "w") as file:
            file.write(request_body)


if __name__ == "__main__":
    get_single_request()
