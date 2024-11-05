import itertools as it
import logging
import multiprocessing as mp
from uuid import UUID

import ipyparallel as ipp
import numpy as np
import pandas as pd
import pyarrow as pa
from lenskit.util import Stopwatch
from progress_api import make_progress

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import ArticleSet
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT, MindData
from poprox_recommender.evaluation.generate.outputs import RecOutputs
from poprox_recommender.lkpipeline import Pipeline
from poprox_recommender.lkpipeline.state import PipelineState
from poprox_recommender.recommenders import recommendation_pipelines

logger = logging.getLogger(__name__)

STAGES = ["final", "ranked", "reranked"]

# globals used for workers
_pipelines: dict[str, Pipeline]
_worker_out: RecOutputs
_emb_seen: set[UUID]


def _init_worker(outs: RecOutputs):
    global _worker_out, _emb_seen, _pipelines
    proc = mp.current_process()
    _worker_out = outs
    _emb_seen = set()

    _worker_out.open(proc.pid)

    _pipelines = recommendation_pipelines(device=default_device())


def _finish_worker():
    logger.info("closing output files")
    _worker_out.close()

    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF)
    except ImportError:
        return None


def _generate_for_request(request: RecommendationRequest) -> UUID | None:
    global _emb_seen

    logger.debug("recommending for user %s", request.interest_profile.profile_id)
    if request.num_recs != TEST_REC_COUNT:
        logger.warning(
            "request for %s had unexpected recommendation count %d",
            request.interest_profile.profile_id,
            request.num_recs,
        )

    pipe_names = list(_pipelines.keys())
    inputs = {
        "candidate": ArticleSet(articles=request.todays_articles),
        "clicked": ArticleSet(articles=request.past_articles),
        "profile": request.interest_profile,
    }

    for name, pipe in _pipelines.items():
        try:
            outputs = pipe.run_all(**inputs)
        except Exception as e:
            logger.error("error recommending for user %s: %s", request.interest_profile.profile_id, e)
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
        assert isinstance(ranked, ArticleSet)
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
        assert isinstance(reranked, ArticleSet)
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

    # get the embeddings
    embedded = pipeline_state.get("candidate-embedder", None)
    embeddings = {}
    if embedded is not None:
        assert isinstance(embedded, ArticleSet)
        assert hasattr(embedded, "embeddings")

        for idx, article in enumerate(embedded.articles):
            embeddings[article.article_id] = embedded.embeddings[idx].cpu().numpy()  # type: ignore
    return output_df, embeddings


def generate_user_recs(dataset: str, outs: RecOutputs, n_users: int | None = None, n_jobs: int = 1):
    mind_data = MindData(dataset)

    logger.info("generating recommendations")

    user_iter = mind_data.iter_users()
    if n_users is None:
        n_users = mind_data.n_users
        logger.info("recommending for all %d users", n_users)
    else:
        logger.info("running on subset of %d users", n_users)
        user_iter = it.islice(user_iter, n_users)

    timer = Stopwatch()
    with make_progress(logger, "recommend", total=n_users) as pb:
        if n_jobs > 1:
            logger.info("starting evaluation with %d workers", n_jobs)
            with ipp.Cluster(n=n_jobs) as client:
                dv = client.direct_view()
                logger.debug("initializing workers")
                dv.apply_sync(_init_worker, outs)

                logger.debug("dispatching jobs")
                lbv = client.load_balanced_view()
                for uid in lbv.imap(_generate_for_request, user_iter, max_outstanding=n_jobs * 5, ordered=False):
                    logger.debug("finished measuring %s", uid)
                    pb.update()

                logger.info("generation finished, closing outputs")
                rusage = dv.apply_sync(_finish_worker)

        else:
            # directly call things in-process
            _init_worker(outs)

            for request in user_iter:
                _generate_for_request(request)
                pb.update()

            _finish_worker()
            rusage = None

    timer.stop()
    logger.info("finished recommending in %s", timer)
    return rusage