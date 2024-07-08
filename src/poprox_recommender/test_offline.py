import logging
import sys

from lenskit.metrics import topn
from tqdm import tqdm

from poprox_recommender.data.mind import MindData

sys.path.append("src")
from uuid import UUID

import numpy as np
import pandas as pd
import torch as th
from safetensors.torch import load_file

from poprox_concepts import ArticleSet
from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_recommender.default import select_articles
from poprox_recommender.paths import model_file_path

logger = logging.getLogger("poprox_recommender.test_offline")


def load_model(device_name=None):
    if device_name is None:
        device_name = "cuda" if th.cuda.is_available() else "cpu"

    load_path = model_file_path("model.safetensors")
    checkpoint = load_file(load_path)

    return checkpoint, device_name


def custom_encoder(obj):
    if isinstance(obj, UUID):
        return str(obj)


def recsys_metric(mind_data: MindData, request: RecommendationRequest, recommendations: ArticleSet):
    # recommendations {account id (uuid): LIST[Article]}
    # use the url of Article

    recs = pd.DataFrame({"item": [a.article_id for a in recommendations.articles]})
    truth = mind_data.user_truth(request.interest_profile.profile_id)

    # RR should look for *clicked* articles, not just all impression articles
    single_rr = topn.recip_rank(recs, truth[truth["rating"] > 0])
    single_ndcg5 = topn.ndcg(recs, truth, k=5)
    single_ndcg10 = topn.ndcg(recs, truth, k=10)

    return single_ndcg5, single_ndcg10, single_rr


if __name__ == "__main__":
    """
    For offline evaluation, set theta in mmr_diversity = 1
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    MODEL, DEVICE = load_model()
    TOKEN_MAPPING = "distilbert-base-uncased"  # can be modified

    mind_data = MindData()

    ngood = 0
    nbad = 0
    ndcg5 = []
    ndcg10 = []
    mrr = []

    logger.info("measuring recommendations")
    for request in tqdm(mind_data.iter_users(), total=mind_data.n_users, desc="recommend"):  # one by one
        logger.debug("recommending for user %s", request.interest_profile.profile_id)
        try:
            recommendations = select_articles(
                ArticleSet(articles=request.todays_articles),
                ArticleSet(articles=request.past_articles),
                request.interest_profile,
                request.num_recs,
            )
        except Exception as e:
            logger.error("error recommending for user %s: %s", request.interest_profile.profile_id, e)
            nbad += 1
            continue

        logger.debug("measuring for user %s", request.interest_profile.profile_id)
        single_ndcg5, single_ndcg10, single_mrr = recsys_metric(mind_data, request, recommendations)
        # recommendations {account id (uuid): LIST[Article]}
        print(
            f"----------------evaluation for {request.interest_profile.profile_id} is NDCG@5 = {single_ndcg5}, NDCG@10 = {single_ndcg10}, RR = {single_mrr}"  # noqa: E501
        )

        ndcg5.append(single_ndcg5)
        ndcg10.append(single_ndcg10)
        mrr.append(single_mrr)
        ngood += 1

    logger.info("recommended for %d users", ngood)
    if nbad:
        logger.error("recommendation FAILED for %d users", nbad)
    print(
        f"Offline evaluation metrics on MIND data: NDCG@5 = {np.mean(ndcg5)}, NDCG@10 = {np.mean(ndcg10)}, MRR = {np.mean(mrr)}"  # noqa: E501
    )

    # response = {"statusCode": 200, "body": json.dump(body, default=custom_encoder)}
