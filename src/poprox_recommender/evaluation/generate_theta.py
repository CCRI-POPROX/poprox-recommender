"""
Generate recommendations for offline test data.

Usage:
    poprox_recommender.evaluation.generate [options] --pipelines=<pipelines>...

Options:
    -v, --verbose
            enable verbose diagnostic logs
    --log-file=FILE
            write log messages to FILE
    -o FILE, --output=FILE
            write output to FILE [default: outputs/recommendations.parquet]
    -M DATA, --mind-data=DATA
            read MIND test data DATA
    --data_path=<data_path>
            path to PopRox data
    --subset=N
            test only on the first N test users
    --pipelines=<pipelines>...
            list of pipeline names (separated by spaces)
"""

# pyright: basic
from __future__ import annotations

import itertools as it
import logging
import logging.config

import hydra
import numpy as np
import pandas as pd
from lenskit.util import Stopwatch
from omegaconf import DictConfig
from progress_api import make_progress

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import CandidateSet, RecommendationList
from poprox_recommender.config import default_device
from poprox_recommender.data.data import Data
from poprox_recommender.data.mind import TEST_REC_COUNT
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.lkpipeline import PipelineState
from poprox_recommender.recommenders import recommendation_pipelines
from poprox_recommender.topics import user_topic_preference

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
                "rank": np.arange(len(recs.articles), dtype=np.int16) + 1,
            }
        )
    ]
    ranked = pipeline_state.get("ranker", None)
    if ranked is not None:
        assert isinstance(ranked, RecommendationList)
        rec_lists.append(
            pd.DataFrame(
                {
                    "recommender": name,
                    "user": str(user),
                    "stage": "ranked",
                    "item": [str(a.article_id) for a in ranked.articles],
                    "rank": np.arange(len(ranked.articles), dtype=np.int16) + 1,
                }
            )
        )
    reranked = pipeline_state.get("reranker", None)
    if reranked is not None:
        assert isinstance(reranked, RecommendationList)
        rec_lists.append(
            pd.DataFrame(
                {
                    "recommender": name,
                    "user": str(user),
                    "stage": "reranked",
                    "item": [str(a.article_id) for a in reranked.articles],
                    "rank": np.arange(len(reranked.articles), dtype=np.int16) + 1,
                }
            )
        )
    output_df = pd.concat(rec_lists, ignore_index=True)
    return output_df


def generate_user_recs(
    data: Data, pipe_names: list[str] | None = None, n_users: int | None = None, theta_topic=0.1, theta_locality=0.1
):
    pipelines = recommendation_pipelines(device=default_device())
    if pipe_names is not None:
        pipelines = {name: pipelines[name] for name in pipe_names}  # type: ignore
    else:
        pipe_names = list(pipelines.keys())

    logger.info("generating recommendations")
    user_recs = []

    user_iter = data.iter_users()
    if n_users is None:
        n_users = data.n_users
        logger.info("recommending for all %d users", n_users)
    else:
        logger.info("running on subset of %d users", n_users)
        user_iter = it.islice(user_iter, n_users)

    timer = Stopwatch()
    with make_progress(logger, "recommend", total=n_users) as pb:
        for request in user_iter:  # one by one
            logger.debug("recommending for user %s", request.interest_profile.profile_id)
            if request.num_recs != TEST_REC_COUNT:
                logger.warn(
                    "request for %s had unexpected recommendation count %d",
                    request.interest_profile.profile_id,
                    request.num_recs,
                )
            # Calculate the clicked topic count
            request.interest_profile.click_topic_counts = user_topic_preference(
                request.past_articles, request.interest_profile.click_history
            )
            inputs = {
                "candidate": CandidateSet(articles=request.todays_articles),
                "clicked": CandidateSet(articles=request.past_articles),
                "profile": request.interest_profile,
                "theta_topic": theta_topic,
                "theta_locality": theta_locality,
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

    timer.stop()
    logger.info("finished recommending in %s", timer)

    return user_recs


@hydra.main(
    config_path="/home/sun00587/research/News_Locality_Polarization/poprox-recommender-locality/src/",
    config_name="config",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    user_recs = generate_user_recs(
        PoproxData(cfg.data_path), cfg.pipelines, theta_topic=cfg.theta_topic, theta_locality=cfg.theta_locality
    )

    print(f"Starting run with theta_topic={round(cfg.theta_topic, 2)}, theta_locality={round(cfg.theta_locality, 2)}")
    all_recs = pd.concat(user_recs, ignore_index=True)
    out_fn = "{}top_{}_loc_{}.parquet".format(cfg.output_dir, round(cfg.theta_topic, 2), round(cfg.theta_locality, 2))
    logger.info("saving recommendations to %s", out_fn)
    all_recs.to_parquet(out_fn, compression="zstd", index=False)


if __name__ == "__main__":
    main()
