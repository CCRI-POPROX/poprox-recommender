import itertools as it
import logging
import multiprocessing as mp
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

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import CandidateSet, RecommendationList
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT
from poprox_recommender.evaluation.generate.outputs import RecOutputs
from poprox_recommender.recommenders import recommendation_pipelines

logger = logging.getLogger(__name__)

STAGES = ["final", "ranked", "reranked"]

# globals used for workers
_pipelines: dict[str, Pipeline]
_worker_out: RecOutputs
_worker_log: WorkerContext | None = None
_emb_seen: set[UUID]


def _init_worker(outs: RecOutputs, logging: WorkerLogConfig | None = None):
    global _worker_out, _emb_seen, _pipelines, _worker_log
    proc = mp.current_process()
    _worker_out = outs
    _emb_seen = set()
    if logging is not None:
        _worker_log = WorkerContext(logging)
        _worker_log.start()

    _worker_out.open(proc.pid)

    _pipelines = recommendation_pipelines(device=default_device())


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


def _generate_for_request(request: RecommendationRequest) -> UUID | None:
    global _emb_seen

    logger.debug("recommending for profile %s", request.interest_profile.profile_id)
    if request.num_recs != TEST_REC_COUNT:
        logger.warning(
            "request for %s had unexpected recommendation count %d",
            request.interest_profile.profile_id,
            request.num_recs,
        )

    pipe_names = list(_pipelines.keys())
    inputs = {
        "candidate": CandidateSet(articles=request.todays_articles),
        "clicked": CandidateSet(articles=request.past_articles),
        "profile": request.interest_profile,
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
    request: RecommendationRequest,
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
                    "profile_id": str(profile),
                    "stage": "ranked",
                    "item_id": [str(a.article_id) for a in ranked.articles],
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
                    "profile_id": str(profile),
                    "stage": "reranked",
                    "item_id": [str(a.article_id) for a in reranked.articles],
                    "rank": np.arange(len(reranked.articles), dtype=np.int16) + 1,
                }
            )
        )
    output_df = pd.concat(rec_lists, ignore_index=True)

    # get the embeddings
    embedded = pipeline_state.get("candidate-embedder", None)
    embeddings = {}
    if embedded is not None:
        assert isinstance(embedded, CandidateSet)
        assert hasattr(embedded, "embeddings")

        for idx, article in enumerate(embedded.articles):
            embeddings[article.article_id] = embedded.embeddings[idx].cpu().numpy()  # type: ignore
    return output_df, embeddings


def generate_profile_recs(dataset: str, outs: RecOutputs, n_profiles: int | None = None, n_jobs: int = 1):
    logger.info("generating recommendations")

    profile_iter = dataset.iter_profiles()
    if n_profiles is None:
        n_profiles = dataset.n_profiles
        logger.info("recommending for all %d profiles", n_profiles)
    else:
        logger.info("running on subset of %d profiles", n_profiles)
        profile_iter = it.islice(profile_iter, n_profiles)

    timer = Stopwatch()
    with item_progress("recommend", total=n_profiles) as pb:
        if n_jobs > 1:
            logger.info("starting evaluation with %d workers", n_jobs)
            with ipp.Cluster(n=n_jobs) as client:
                dv = client.direct_view()
                logger.debug("initializing workers")
                dv.apply_sync(_init_worker, outs, WorkerLogConfig.current())

                logger.debug("dispatching jobs")
                lbv = client.load_balanced_view()
                for uid in lbv.imap(_generate_for_request, profile_iter, max_outstanding=n_jobs * 5, ordered=False):
                    logger.debug("finished measuring %s", uid)
                    pb.update()

                logger.info("generation finished, closing outputs")
                rusage = dv.apply_sync(_finish_worker)

        else:
            logger.info("starting serial evaluation")
            # directly call things in-process
            _init_worker(outs)

            for request in profile_iter:
                _generate_for_request(request)
                pb.update()

            _finish_worker()
            rusage = None

    timer.stop()
    logger.info("finished recommending in %s", timer)
    return rusage
