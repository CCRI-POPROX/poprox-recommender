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
from progress_api import make_progress

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import ArticleSet
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT, MindData
from poprox_recommender.lkpipeline import PipelineState
from poprox_recommender.logging_config import setup_logging
from poprox_recommender.recommenders import recommendation_pipelines

logger = logging.getLogger("poprox_recommender.evaluation.evaluate")

# long-term TODO:
# - support other MIND data (test?)
# - support our data

STAGES = ["final", "ranked", "reranked"]


def extract_recs(
    name: str,
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
                "recommender": name,
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
                    "recommender": name,
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
                    "recommender": name,
                    "user": str(user),
                    "stage": "reranked",
                    "item": [str(a.article_id) for a in reranked.articles],
                }
            )
        )
    output_df = pd.concat(rec_lists, ignore_index=True)
    return output_df


def generate_user_recs():
    mind_data = MindData()

    pipelines = recommendation_pipelines(device=default_device())
    pipe_names = list(pipelines.keys())

    logger.info("generating recommendations")
    user_recs = []

    with make_progress(logger, "recommend", total=mind_data.n_users) as pb:
        for request in mind_data.iter_users():  # one by one
            logger.debug("recommending for user %s", request.interest_profile.profile_id)
            if request.num_recs != TEST_REC_COUNT:
                logger.warn(
                    "request for %s had unexpected recommendation count %d",
                    request.interest_profile.profile_id,
                    request.num_recs,
                )
            inputs = {
                "candidate": ArticleSet(articles=request.todays_articles),
                "clicked": ArticleSet(articles=request.past_articles),
                "profile": request.interest_profile,
            }
            for name, pipe in pipelines.items():
                try:
                    outputs = pipe.run_all(**inputs)
                except Exception as e:
                    logger.error("error recommending for user %s: %s", request.interest_profile.profile_id, e)
                    raise e
                user_df = extract_recs(name, request, outputs)
                user_df["recommender"] = pd.Categorical(user_df["recommender"], categories=pipe_names)
                user_df["stage"] = pd.Categorical(user_df["stage"].astype("category"), categories=STAGES)
                user_recs.append(user_df)
            pb.update()

    return user_recs


if __name__ == "__main__":
    """
    For offline evaluation, set theta in mmr_diversity = 1
    """
    options = docopt(__doc__)  # type: ignore
    setup_logging(verbose=options["--verbose"], log_file=options["--log-file"])

    user_recs = generate_user_recs()

    all_recs = pd.concat(user_recs, ignore_index=True)
    out_fn = options["--output"]
    logger.info("saving recommendations to %s", out_fn)
    all_recs.to_parquet(out_fn, compression="zstd", index=False)

    # response = {"statusCode": 200, "body": json.dump(body, default=custom_encoder)}
