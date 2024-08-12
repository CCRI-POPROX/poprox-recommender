"""
Generate recommendations for offline test data.

Usage:
    poprox_recommender.evaluation.generate [options]

Options:
    -v, --verbose
            enable verbose diagnostic logs
    --log-file=FILE
            write log messages to FILE
    -o FILE, --output=FILE
            write output to FILE [default: outputs/recommendations.parquet]
"""

# pyright: basic
from __future__ import annotations

import logging
import logging.config

import pandas as pd
from docopt import docopt
from tqdm import tqdm

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import ArticleSet
from poprox_recommender.data.mind import TEST_REC_COUNT, MindData
from poprox_recommender.default import personalized_pipeline
from poprox_recommender.lkpipeline import PipelineState
from poprox_recommender.logging_config import setup_logging

logger = logging.getLogger(__name__)

# long-term TODO:
# - support other MIND data (test?)
# - support our data


def extract_recs(
    request: RecommendationRequest,
    pipeline_state: PipelineState,
) -> pd.DataFrame:
    # recommendations {account id (uuid): LIST[Article]}
    # use the url of Article
    user = request.interest_profile.profile_id
    assert user is not None

    # get the different recommendation lists to record
    recs = pipeline_state["recommender"]
    rec_lists = [
        pd.DataFrame(
            {
                "user": str(user),
                "stage": "final",
                "item": [str(a.article_id) for a in recs.articles],
            }
        )
    ]
    ranked = pipeline_state.get("ranker", None)
    if ranked is not None:
        assert isinstance(ranked, ArticleSet)
        rec_lists.append(
            pd.DataFrame(
                {
                    "user": str(user),
                    "stage": "ranked",
                    "item": [str(a.article_id) for a in ranked.articles],
                }
            )
        )
    reranked = pipeline_state.get("reranker", None)
    if reranked is not None:
        assert isinstance(reranked, ArticleSet)
        rec_lists.append(
            pd.DataFrame(
                {
                    "user": str(user),
                    "stage": "reranked",
                    "item": [str(a.article_id) for a in reranked.articles],
                }
            )
        )
    output_df = pd.concat(rec_lists, ignore_index=True)
    # make stage a categorical to save memory + disk
    output_df["stage"] = output_df["stage"] = pd.Categorical(
        output_df["stage"], categories=["final", "ranked", "reranked"]
    )
    return output_df


def generate_user_recs():
    mind_data = MindData()

    pipeline = personalized_pipeline(TEST_REC_COUNT)
    assert pipeline is not None

    logger.info("generating recommendations")
    user_recs = []

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
            outputs = pipeline.run_all("recommender", **inputs)
        except Exception as e:
            logger.error("error recommending for user %s: %s", request.interest_profile.profile_id, e)
            raise e
        user_recs.append(extract_recs(request, outputs))
    return user_recs


if __name__ == "__main__":
    """
    For offline evaluation, set theta in mmr_diversity = 1
    """
    options = docopt(__doc__)
    setup_logging(verbose=options["--verbose"], log_file=options["--log-file"])

    user_recs = generate_user_recs()

    all_recs = pd.concat(user_recs, ignore_index=True)
    out_fn = options["--output"]
    logger.info("saving recommendations to %s", out_fn)
    all_recs.to_parquet(out_fn, compression="zstd", index=False)

    # response = {"statusCode": 200, "body": json.dump(body, default=custom_encoder)}
