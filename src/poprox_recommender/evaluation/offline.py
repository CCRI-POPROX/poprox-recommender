import csv
import gzip
import json
import logging
import logging.config
import sys

from lenskit.metrics import topn
from tqdm import tqdm

from poprox_recommender.data.mind import TEST_REC_COUNT, MindData
from poprox_recommender.lkpipeline import PipelineState
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.pipeline import RecommendationPipeline

sys.path.append("src")
from uuid import UUID

import numpy as np
import pandas as pd
import torch as th
from safetensors.torch import load_file

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import ArticleSet
from poprox_recommender.evaluation.metrics import rank_biased_overlap
from poprox_recommender.paths import model_file_path, project_root
from poprox_recommender.recommenders import recommendation_pipelines

logger = logging.getLogger("poprox_recommender.test_offline")


def load_model(device_name=None):
    if device_name is None:
        device_name = "cuda" if th.cuda.is_available() else "cpu"

    load_path = model_file_path("nrms-mind/model.safetensors")
    checkpoint = load_file(load_path)

    return checkpoint, device_name


def custom_encoder(obj):
    if isinstance(obj, UUID):
        return str(obj)


def recsys_metric(mind_data: MindData, request: RecommendationRequest, state: PipelineState):
    # recommendations {account id (uuid): LIST[Article]}
    # use the url of Article
    final = state["recommender"]

    recs = pd.DataFrame({"item": [a.article_id for a in final.articles]})
    truth = mind_data.user_truth(request.interest_profile.profile_id)

    # RR should look for *clicked* articles, not just all impression articles
    single_rr = topn.recip_rank(recs, truth[truth["rating"] > 0])
    single_ndcg5 = topn.ndcg(recs, truth, k=5)
    single_ndcg10 = topn.ndcg(recs, truth, k=10)

    if "ranker" in state:
        topk = state["ranker"]
        single_rbo5 = rank_biased_overlap(topk, final, k=5)
        single_rbo10 = rank_biased_overlap(topk, final, k=10)
    else:
        single_rbo5 = None
        single_rbo10 = None

    return single_ndcg5, single_ndcg10, single_rr, single_rbo5, single_rbo10


if __name__ == "__main__":
    """
    For offline evaluation, set theta in mmr_diversity = 1
    """
    setup_logging(log_file="eval.log")

    MODEL, DEVICE = load_model()
    TOKEN_MAPPING = "distilbert-base-uncased"  # can be modified

    mind_data = MindData()

    ngood = 0
    nbad = 0
    ndcg5 = []
    ndcg10 = []
    recip_rank = []
    rbo5 = []
    rbo10 = []

    pipeline: RecommendationPipeline = recommendation_pipelines(num_slots=TEST_REC_COUNT)["nrms"]

    logger.info("measuring recommendations")
    user_out_fn = project_root() / "outputs" / "user-metrics.csv.gz"
    user_out_fn.parent.mkdir(exist_ok=True, parents=True)
    user_out = gzip.open(user_out_fn, "wt")
    user_csv = csv.writer(user_out)
    user_csv.writerow(["user_id", "personalized", "NDCG@5", "NDCG@10", "RecipRank", "RBO@5", "RBO@10"])

    for request in tqdm(mind_data.iter_users(), total=mind_data.n_users, desc="recommend"):  # one by one
        logger.debug("recommending for user %s", request.interest_profile.profile_id)
        if request.num_recs != TEST_REC_COUNT:
            logger.warn(
                "request for %s had unexpected recommendation count %d",
                request.interest_profile.profile_id,
                request.num_recs,
            )
        try:
            inputs = {
                "candidate": ArticleSet(articles=request.todays_articles),
                "clicked": ArticleSet(articles=request.past_articles),
                "profile": request.interest_profile,
            }
            state = pipeline.run_all(**inputs)
            if request.interest_profile.click_history.article_ids:
                personalized = 1
            else:
                personalized = 0
        except Exception as e:
            logger.error("error recommending for user %s: %s", request.interest_profile.profile_id, e)
            raise e

        logger.debug("measuring for user %s", request.interest_profile.profile_id)
        single_ndcg5, single_ndcg10, single_rr, single_rbo5, single_rbo10 = recsys_metric(mind_data, request, state)
        user_csv.writerow(
            [
                request.interest_profile.profile_id,
                personalized,
                single_ndcg5,
                single_ndcg10,
                single_rr,
                single_rbo5,
                single_rbo10,
            ]
        )
        # recommendations {account id (uuid): LIST[Article]}
        logger.debug(
            "user %s: NDCG@5=%0.3f, NDCG@10=%0.3f, RR=%0.3f, RBO@5=%0.3f, RBO@10=%0.3f",
            request.interest_profile.profile_id,
            single_ndcg5,
            single_ndcg10,
            single_rr,
            single_rbo5 or -1.0,
            single_rbo10 or -1.0,
        )

        ndcg5.append(single_ndcg5)
        ndcg10.append(single_ndcg10)
        recip_rank.append(single_rr)
        rbo5.append(single_rbo5 or np.nan)
        rbo10.append(single_rbo10 or np.nan)
        ngood += 1

    user_out.close()

    logger.info("recommended for %d users", ngood)
    if nbad:
        logger.error("recommendation FAILED for %d users", nbad)
    agg_metrics = {
        "NDCG@5": np.mean(ndcg5),
        "NDCG@10": np.mean(ndcg10),
        "MRR": np.mean(recip_rank),
        "RBO@5": np.nanmean(rbo5),
        "RBO@10": np.nanmean(rbo10),
    }
    out_fn = project_root() / "outputs" / "metrics.json"
    out_fn.parent.mkdir(exist_ok=True, parents=True)
    out_fn.write_text(json.dumps(agg_metrics) + "\n")
    logger.info("Mean NDCG@5: %.3f", np.mean(ndcg5))
    logger.info("Mean NDCG@10: %.3f", np.mean(ndcg10))
    logger.info("Mean RR: %.3f", np.mean(recip_rank))
    logger.info("Mean RBO@5: %.3f", np.mean(rbo5))
    logger.info("Mean RBO@10: %.3f", np.mean(rbo10))

    # response = {"statusCode": 200, "body": json.dump(body, default=custom_encoder)}
