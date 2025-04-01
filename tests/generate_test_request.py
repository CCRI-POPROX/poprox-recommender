"""
With exported dataset, this script can help generate test request body for recommender pipieline testing
The data export dataset needs to be placed in side poprox-recommender/data/POPROX folder, with timestamp removed

Usage:
    generate_test_request.py [--account_id ID] [--min_click_num NUM] [--output_file OUTPUT]

Options:
    --account_id ID      Specific user account id to process request data
    --min_click_num NUM      Get a random user with at least min_click_num clicks
    --output_file OUTPUT Path to the output file
"""

import json
import random

from docopt import docopt

from poprox_concepts.api.recommendations import RecommendationRequestV2
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.paths import project_root


def get_single_request() -> str:
    options = docopt(__doc__)
    eval_data = PoproxData()
    requests = list(eval_data.iter_profiles())
    excluded_fields = {"__all__": {"raw_data": True, "images": {"__all__": {"raw_data"}}}}

    request_body = ""
    if options["--account_id"]:
        account_id = options["--account_id"]
        for req in requests:
            if req.interest_profile.profile_id == account_id:
                request_body = RecommendationRequestV2.model_dump_json(
                    req,
                    exclude={"candidates": excluded_fields, "interacted": excluded_fields, "protocol_version": True},
                )
                break
    elif options["--min_click_num"]:
        min_click = options["--min_click_num"]
        for req in requests:
            print(len(req.interest_profile.click_history))
            if req.interest_profile.click_history and len(req.interest_profile.click_history) >= int(min_click):
                request_body = RecommendationRequestV2.model_dump_json(
                    req,
                    exclude={"candidates": excluded_fields, "interacted": excluded_fields, "protocol_version": True},
                )
                break
    else:
        random_index = random.randint(0, len(requests) - 1)
        request_body = RecommendationRequestV2.model_dump_json(
            requests[random_index],
            exclude={"candidates": excluded_fields, "interacted": excluded_fields, "protocol_version": True},
        )

    if options["--output_file"]:
        with open(options["--output_file"], "w") as file:
            file.write(request_body)
    else:
        request_data_path = project_root() / "tests" / "request_data" / "request_body_1.json"
        with open(request_data_path, "w") as file:
            file.write(request_body)


if __name__ == "__main__":
    get_single_request()
