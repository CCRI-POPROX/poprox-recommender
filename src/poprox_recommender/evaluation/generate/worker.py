import itertools as it
import logging
import multiprocessing as mp
from typing import Tuple
from uuid import UUID

import ipyparallel as ipp
import numpy as np
import pandas as pd
import pyarrow as pa
from lenskit.logging import item_progress
from lenskit.logging.worker import WorkerContext, WorkerLogConfig
from lenskit.pipeline import Pipeline
from lenskit.pipeline.state import PipelineState
from lenskit.util import Stopwatch

from poprox_concepts.api.recommendations import RecommendationRequestV2
from poprox_concepts.domain import CandidateSet, RecommendationList
from poprox_recommender.components.diversifiers.locality_calibration import LocalityCalibratorConfig
from poprox_recommender.components.generators.context import ContextGeneratorConfig
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT
from poprox_recommender.data.poprox import PoproxData
from poprox_recommender.evaluation.generate.outputs import RecOutputs
from poprox_recommender.recommenders import load_all_pipelines
from poprox_recommender.topics import user_topic_preference

logger = logging.getLogger(__name__)

STAGES = ["final", "ranked", "reranked", "generator"]

THETA_RANDOM_SAMPLES = 60

# globals used for workers
_pipelines: dict[str, Pipeline]
_worker_out: RecOutputs
_worker_log: WorkerContext | None = None
_emb_seen: set[UUID]


def _init_worker(outs: RecOutputs, logging: WorkerLogConfig | None = None, pipelines: list[str] | None = None):
    global _worker_out, _emb_seen, _pipelines, _worker_log
    try:
        import asyncio

        asyncio.get_event_loop().close()
    except Exception:
        pass

    proc = mp.current_process()
    _worker_out = outs
    _emb_seen = set()
    if logging is not None:
        _worker_log = WorkerContext(logging)
        _worker_log.start()

    _worker_out.open(proc.pid)

    _pipelines = load_all_pipelines(device=default_device())
    if pipelines:
        logger.warning(f"Subsetting {_pipelines} to {pipelines}...")
        # _pipelines = {k: v for k, v in _pipelines.items() if k in pipelines}
        _pipelines = {name: _pipelines[name] for name in pipelines}


def _finish_worker():
    global _worker_log
    logger.info("closing output files")
    _worker_out.close()
    if _worker_log is not None:
        _worker_log.shutdown()
        _worker_log = None

    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF)
    except ImportError:
        return None


def _generate_for_request(request: RecommendationRequestV2) -> UUID | None:
    return _generate_for_hyperparamter_request((request, (None, None)))


def _generate_for_hyperparamter_request_threshold(
    request_with_threshold: Tuple[RecommendationRequestV2, float | None],
) -> UUID | None:
    global _emb_seen

    request = request_with_threshold[0]
    threshold = request_with_threshold[1]

    logger.debug("recommending for profile %s", request.interest_profile.profile_id)
    if request.num_recs != TEST_REC_COUNT:
        logger.warning(
            "request for %s had unexpected recommendation count %d",
            request.interest_profile.profile_id,
            request.num_recs,
        )

    pipe_names = list(_pipelines.keys())
    logger.warning(f"Generating hyperparameter request for {pipe_names}...")

    # Calculate the clicked topic count
    request.interest_profile.click_topic_counts = user_topic_preference(
        request.interacted.articles, request.interest_profile.click_history
    )

    inputs = {
        "candidate": request.candidates,
        "clicked": request.interacted,
        "profile": request.interest_profile,
        "similarity_threshold": threshold,
    }

    for name, pipe in _pipelines.items():
        try:
            outputs = pipe.run_all(**inputs)
        except Exception as e:
            logger.error("error recommending for profile %s: %s", request.interest_profile.profile_id, e)
            raise e

        rec_df, embeddings = extract_recs(name, request, outputs)
        rec_df["recommender"] = pd.Categorical(rec_df["recommender"], categories=pipe_names)
        rec_df["stage"] = pd.Categorical(rec_df["stage"].astype("category"), categories=STAGES)
        rec_df["similarity_threshold"] = threshold
        rec_df["theta_topic"] = LocalityCalibratorConfig().theta_topic
        rec_df["theta_locality"] = LocalityCalibratorConfig().theta_locality
        _worker_out.rec_writer.write_frame(rec_df)

        # find any embeddings not yet written
        emb_rows = [
            {"article_id": str(aid), "embedding": emb} for (aid, emb) in embeddings.items() if aid not in _emb_seen
        ]
        _emb_seen |= embeddings.keys()
        if emb_rows:
            # directly use pyarrow to avoid DF overhead, small but easy to avoid here
            emb_tbl = pa.Table.from_pylist(emb_rows)
            _worker_out.emb_writer.write_frame(emb_tbl)

    # just return the ID to indicate success
    return request.interest_profile.profile_id


def _generate_for_hyperparamter_request(
    request_with_thetas: Tuple[RecommendationRequestV2, Tuple[float | None, float | None]],
) -> UUID | None:
    global _emb_seen

    request = request_with_thetas[0]
    thetas = request_with_thetas[1]

    logger.debug("recommending for profile %s", request.interest_profile.profile_id)
    if request.num_recs != TEST_REC_COUNT:
        logger.warning(
            "request for %s had unexpected recommendation count %d",
            request.interest_profile.profile_id,
            request.num_recs,
        )

    pipe_names = list(_pipelines.keys())
    logger.warning(f"Generating hyperparameter request for {pipe_names}...")

    # Calculate the clicked topic count
    request.interest_profile.click_topic_counts = user_topic_preference(
        request.interacted.articles, request.interest_profile.click_history
    )

    inputs = {
        "candidate": request.candidates,
        "clicked": request.interacted,
        "profile": request.interest_profile,
        "theta_topic": thetas[0],
        "theta_locality": thetas[1],
    }

    for name, pipe in _pipelines.items():
        try:
            outputs = pipe.run_all(**inputs)
        except Exception as e:
            logger.error("error recommending for profile %s: %s", request.interest_profile.profile_id, e)
            raise e

        rec_df, embeddings = extract_recs(name, request, outputs)
        rec_df["recommender"] = pd.Categorical(rec_df["recommender"], categories=pipe_names)
        rec_df["stage"] = pd.Categorical(rec_df["stage"].astype("category"), categories=STAGES)
        rec_df["similarity_threshold"] = ContextGeneratorConfig().similarity_threshold
        rec_df["theta_topic"] = thetas[0]
        rec_df["theta_locality"] = thetas[1]
        _worker_out.rec_writer.write_frame(rec_df)

        # find any embeddings not yet written
        emb_rows = [
            {"article_id": str(aid), "embedding": emb} for (aid, emb) in embeddings.items() if aid not in _emb_seen
        ]
        _emb_seen |= embeddings.keys()
        if emb_rows:
            # directly use pyarrow to avoid DF overhead, small but easy to avoid here
            emb_tbl = pa.Table.from_pylist(emb_rows)
            _worker_out.emb_writer.write_frame(emb_tbl)

    # just return the ID to indicate success
    return request.interest_profile.profile_id


def extract_recs(
    name: str,
    request: RecommendationRequestV2,
    pipeline_state: PipelineState,
) -> tuple[pd.DataFrame, dict[UUID, np.ndarray]]:
    # recommendations {account id (uuid): LIST[Article]}
    # use the url of Article
    profile = request.interest_profile.profile_id
    assert profile is not None

    # get the different recommendation lists to record
    recs = pipeline_state["recommender"]
    rec_lists = [
        pd.DataFrame(
            {
                "recommender": name,
                "profile_id": str(profile),
                "stage": "final",
                "item_id": [str(a.article_id) for a in recs.articles],
                "rank": np.arange(len(recs.articles), dtype=np.int16) + 1,
                "treatment": 0.0,
                "total_prompts": 0.0,
                "prompt_level_ratio": 0.0,
                "k1_topic": -1.0,
                "k1_locality": -1.0,
                "is_inside_locality_threshold": False,
                "prompt_level": "NA",
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
            }
        )
    ]
    ranked = pipeline_state.get("ranker", None)
    if ranked is not None:
        assert isinstance(ranked, RecommendationList), f"reranked has unexpected type {type(ranked)} in pipeline {name}"
        rec_lists.append(
            pd.DataFrame(
                {
                    "recommender": name,
                    "profile_id": str(profile),
                    "stage": "ranked",
                    "item_id": [str(a.article_id) for a in ranked.articles],
                    "rank": np.arange(len(ranked.articles), dtype=np.int16) + 1,
                    "treatment": 0.0,
                    "total_prompts": 0.0,
                    "prompt_level_ratio": 0.0,
                    "k1_topic": -1.0,
                    "k1_locality": -1.0,
                    "is_inside_locality_threshold": False,
                    "prompt_level": "NA",
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                }
            )
        )
    reranked = pipeline_state.get("reranker", None)
    if reranked is not None:
        assert isinstance(reranked, CandidateSet), f"reranked has unexpected type {type(reranked)} in pipeline {name}"
        rec_lists.append(
            pd.DataFrame(
                {
                    "recommender": name,
                    "profile_id": str(profile),
                    "stage": "reranked",
                    "item_id": [str(a.article_id) for a in reranked.articles],
                    "rank": np.arange(len(reranked.articles), dtype=np.int16) + 1,
                    "treatment": reranked.treatment_flags,  # type: ignore
                    "total_prompts": 0.0,
                    "prompt_level_ratio": 0.0,
                    "k1_topic": reranked.k1_topic,
                    "k1_locality": reranked.k1_locality,
                    "is_inside_locality_threshold": reranked.is_inside_locality_threshold,
                    "prompt_level": "NA",
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                }
            )
        )
    generator = pipeline_state.get("generator", None)
    if generator is not None:
        assert isinstance(generator, RecommendationList)

        num_event_level_prompts = 0.0
        num_topic_level_prompts = 0.0
        for news_extra in generator.extras:
            if news_extra.get("prompt_level") == "event":
                num_event_level_prompts += 1
            elif news_extra.get("prompt_level") == "topic":
                num_topic_level_prompts += 1
        total_prompts = num_event_level_prompts + num_topic_level_prompts
        rec_lists.append(
            pd.DataFrame(
                {
                    "recommender": name,
                    "profile_id": str(profile),
                    "stage": "generator",
                    "item_id": [str(a.article_id) for a in generator.articles],
                    "rank": np.arange(len(generator.articles), dtype=np.int16) + 1,
                    "treatment": 0.0,
                    "total_prompts": total_prompts,
                    "prompt_level_ratio": 0.0
                    if total_prompts == 0
                    else num_event_level_prompts
                    / total_prompts,  # event-level / treatment number ( 1 = best, 0 = worst aka all topic-level)
                    "k1_topic": -1.0,
                    "k1_locality": -1.0,
                    "is_inside_locality_threshold": False,
                    "prompt_level": [extra.get("prompt_level", None) for extra in generator.extras],
                    "rouge1": [extra.get("rouge1", None) for extra in generator.extras],
                    "rouge2": [extra.get("rouge2", None) for extra in generator.extras],
                    "rougeL": [extra.get("rougeL", None) for extra in generator.extras],
                }
            )
        )
    output_df = pd.concat(rec_lists, ignore_index=True)

    # get the embeddings
    embedded = pipeline_state.get("candidate-embedder", None)
    embeddings = {}
    if embedded is not None:
        assert isinstance(embedded, CandidateSet), f"embedded has unexpected type {type(embedded)} in pipeline {name}"
        assert hasattr(embedded, "embeddings")

        for idx, article in enumerate(embedded.articles):
            embeddings[article.article_id] = embedded.embeddings[idx].cpu().numpy()  # type: ignore
    return output_df, embeddings


def create_cluster(n_jobs):
    """Create cluster with proper cleanup handling"""
    # Cleanup any existing clusters first
    try:
        ipp.Client().close()
    except Exception:
        pass

    cluster = ipp.Cluster(
        n=n_jobs,
        reuse_engines=False,  # Fresh engines each time
    )
    cluster.start_cluster_sync()
    return cluster


def generate_profile_recs(
    dataset: PoproxData,
    outs: RecOutputs,
    n_profiles: int | None = None,
    n_jobs: int = 1,
    topic_thetas: tuple[float, float] | None = None,
    locality_thetas: tuple[float, float] | None = None,
    similarity_thresholds: tuple[float, float] | None = None,
    pipelines: list[str] | None = None,
):
    logger.info(f"generating recommendations for pipelines {pipelines}")

    if topic_thetas and locality_thetas:
        profile_iter = dataset.iter_hyperparameters_theta(
            topic_thetas, 0.05, locality_thetas, 0.05, random_sample=THETA_RANDOM_SAMPLES
        )
    elif similarity_thresholds:
        profile_iter = dataset.iter_hyperparameters_theshold(similarity_thresholds, 0.05)
    else:
        profile_iter = dataset.iter_profiles()

    if n_profiles is None:
        n_profiles = dataset.n_profiles
        logger.info("recommending for all %d profiles", n_profiles)
    else:
        logger.info("running on subset of %d profiles", n_profiles)
        profile_iter = it.islice(profile_iter, n_profiles)

    if topic_thetas and locality_thetas:
        n_profiles = n_profiles * dataset.n_hyperparameters

    timer = Stopwatch()
    with item_progress("recommend", total=n_profiles) as pb:
        if n_jobs > 1:
            logger.info("starting evaluation with %d workers", n_jobs)
            try:
                with create_cluster(n_jobs) as client:
                    dv = client.direct_view()
                    logger.debug("initializing workers")
                    dv.apply_sync(_init_worker, outs, WorkerLogConfig.current(), pipelines)

                    logger.debug("dispatching jobs")
                    lbv = client.load_balanced_view()
                    if topic_thetas and locality_thetas:
                        request_iter = lbv.imap(
                            _generate_for_hyperparamter_request,
                            profile_iter,
                            max_outstanding=n_jobs * 5,
                            ordered=False,
                        )
                    elif similarity_thresholds:
                        request_iter = lbv.imap(
                            _generate_for_hyperparamter_request_threshold,
                            profile_iter,
                            max_outstanding=n_jobs * 5,
                            ordered=False,
                        )
                    else:
                        request_iter = lbv.imap(
                            _generate_for_request,
                            profile_iter,
                            max_outstanding=n_jobs * 5,
                            ordered=False,
                        )

                    for uid in request_iter:
                        logger.debug("finished measuring %s", uid)
                        pb.update()

                    logger.info("generation finished, closing outputs")
                    rusage = dv.apply_sync(_finish_worker)
            finally:
                # Ensure cleanup
                ipp.Client().purge_everything()

        else:
            logger.info(f"starting serial evaluation for pipelines: {pipelines}")
            # directly call things in-process
            _init_worker(outs, pipelines=pipelines)

            for request in profile_iter:
                if topic_thetas and locality_thetas:
                    _generate_for_hyperparamter_request(request)
                elif similarity_thresholds:
                    _generate_for_hyperparamter_request_threshold(request)
                else:
                    _generate_for_request(request)
                pb.update()

            _finish_worker()
            rusage = None

    timer.stop()
    logger.info("finished recommending in %s", timer)
    return rusage
