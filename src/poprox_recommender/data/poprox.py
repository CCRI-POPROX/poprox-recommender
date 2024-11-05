"""
Support for loading POPROX data for evaluation.

"""

import json
import logging
import os
from typing import Generator

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.data.data import Data

logger = logging.getLogger(__name__)
TEST_REC_COUNT = 10


class PoproxData(Data):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    @property
    def n_users(self) -> int:
        # Count the number of JSON files in the folder path
        return sum(1 for filename in os.listdir(self.folder_path) if filename.endswith(".json"))

    def iter_users(self) -> Generator[RecommendationRequest, None, None]:
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.folder_path, filename)

                with open(file_path, "r") as file:
                    data = json.load(file)

                # yield RecommendationRequest.model_validate_json(data)
                yield RecommendationRequest(
                    todays_articles=data.get("todays_articles"),
                    past_articles=data.get("past_articles"),
                    interest_profile=data.get("interest_profile"),
                    num_recs=data.get("num_recs"),
                )
