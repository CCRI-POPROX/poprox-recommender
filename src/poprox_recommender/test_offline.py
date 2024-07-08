import logging
import sys

from lenskit.metrics import topn
from tqdm import tqdm

from poprox_recommender.data.mind import load_mind_data

sys.path.append("src")
from uuid import UUID

import numpy as np
import pandas as pd
import torch as th
from safetensors.torch import load_file

from poprox_concepts import Article, ArticleSet, ClickHistory, InterestProfile
from poprox_recommender.default import select_articles
from poprox_recommender.paths import model_file_path, project_root

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


def recsys_metric(recommendations: ArticleSet, row_index, news_struuid_ID: dict[str, str]):
    # recommendations {account id (uuid): LIST[Article]}
    # use the url of Article
    ## FIXME: we are reloading this for *every* user, that's really slow
    impressions_truth = (
        pd.read_table(
            project_root() / "data" / "test_mind_large" / "behaviors.tsv",
            header="infer",
            usecols=range(5),
            names=["impression_id", "user", "time", "clicked_news", "impressions"],
        )
        .iloc[row_index]
        .impressions.split()
    )  # LIST[ID-label]

    recommended_list = [news_struuid_ID[item.url] for item in recommendations.articles]

    recs = pd.DataFrame({"item": recommended_list})

    truth = pd.DataFrame.from_records(
        ((row.split("-")[0], int(row.split("-")[1])) for row in impressions_truth), columns=["item", "rating"]
    ).set_index("item")

    single_rr = topn.recip_rank(recs, truth)
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

    mind_data = load_mind_data()

    ngood = 0
    nbad = 0
    ndcg5 = []
    ndcg10 = []
    mrr = []

    logger.info("measuring recommendations")
    for impression_idx in tqdm(range(10), desc="recommend"):  # one by one
        request_body = mind_data.test_list[impression_idx]

        todays_articles = [Article.parse_obj(item) for item in request_body["todays_articles"]]
        past_articles = [Article.parse_obj(item) for item in request_body["past_articles"]]

        click_data = ClickHistory.parse_obj(request_body["click_data"])
        profile = InterestProfile(profile_id=click_data.account_id, click_history=click_data, onboarding_topics=[])

        logger.debug("recommending for user %s", profile.profile_id)
        try:
            recommendations = select_articles(
                ArticleSet(articles=todays_articles),
                ArticleSet(articles=past_articles),
                profile,
                request_body["num_recs"],
            )
        except Exception as e:
            logger.error("error recommending for user %s: %s", profile.profile_id, e)
            nbad += 1
            continue

        logger.debug("measuring for user %s", profile.profile_id)
        single_ndcg5, single_ndcg10, single_mrr = recsys_metric(recommendations, impression_idx, mind_data.news_uuid_ID)
        # recommendations {account id (uuid): LIST[Article]}
        print(
            f"----------------evaluation using the first {impression_idx + 1} is NDCG@5 = {single_ndcg5}, NDCG@10 = {single_ndcg10}, RR = {single_mrr}"  # noqa: E501
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
