import itertools as it
import multiprocessing as mp
from uuid import UUID

import numpy as np
import pandas as pd
import ray
import torch
from lenskit.logging import get_logger, item_progress
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import init_cluster
from lenskit.pipeline import Pipeline
from lenskit.pipeline.state import PipelineState
from lenskit.util import Stopwatch

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import CandidateSet, RecommendationList
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT
from poprox_recommender.evaluation.generate.outputs import EmbeddingWriter, RecOutputs
from poprox_recommender.recommenders import load_all_pipelines

logger = get_logger(__name__)

BATCH_SIZE = 10
STAGES = ["final", "ranked", "reranked"]


class RecGenerator:
    """
    Generate recommendations. Can be used as a Ray actor.
    """

    pipelines: dict[str, Pipeline]
    worker_out: RecOutputs
    emb_seen: set[UUID]
    emb_writer: EmbeddingWriter

    def __init__(self, outs: RecOutputs, emb_writer: EmbeddingWriter):
        proc = mp.current_process()
        self.pipelines = load_all_pipelines(device=default_device())
        self.worker_out = outs
        self.worker_out.open(proc.pid)
        self.emb_seen = set()
        self.emb_writer = emb_writer

    def generate(self, request: RecommendationRequest) -> UUID | None:
        log = logger.bind(profile_id=str(request.interest_profile.profile_id))
        log.debug("beginning recommendation")
        if request.num_recs != TEST_REC_COUNT:
            log.warning(
                "request for %s had unexpected recommendation count %d",
                request.interest_profile.profile_id,
                request.num_recs,
            )

        pipe_names = list(self.pipelines.keys())
        inputs = {
            "candidate": CandidateSet(articles=request.todays_articles),
            "clicked": CandidateSet(articles=request.past_articles),
            "profile": request.interest_profile,
        }

        for name, pipe in self.pipelines.items():
            try:
                outputs = pipe.run_all(**inputs)
            except Exception as e:
                logger.error("error recommending for profile %s: %s", request.interest_profile.profile_id, e)
                raise e

            rec_df, embeddings = extract_recs(name, request, outputs)
            rec_df["recommender"] = pd.Categorical(rec_df["recommender"], categories=pipe_names)
            rec_df["stage"] = pd.Categorical(rec_df["stage"].astype("category"), categories=STAGES)
            self.worker_out.rec_writer.write_frame(rec_df)

            # find any embeddings we haven't yet written (reduces overhead)
            # the writer will also deduplicate between workers.
            emb_to_write = {aid: emb for (aid, emb) in embeddings.items() if aid not in self.emb_seen}
            # call remote if we have an actor
            if hasattr(self.emb_writer.write_embeddings, "remote"):
                ray.get(self.emb_writer.write_embeddings.remote(emb_to_write))
            else:
                self.emb_writer.write_embeddings(emb_to_write)
            self.emb_seen |= embeddings.keys()

        # just return the ID to indicate success
        return request.interest_profile.profile_id

    def generate_batch(self, batch: list[RecommendationRequest]) -> list[UUID | None]:
        return [self.generate(req) for req in batch]

    def finish(self):
        self.worker_out.close()

        try:
            import resource

            return resource.getrusage(resource.RUSAGE_SELF)
        except ImportError:
            return None


def dynamic_remote(actor):
    pc = get_parallel_config()
    if torch.cuda.is_available():
        _cuda_props = torch.cuda.get_device_properties()
        # Let's take a wild guess that 20 MP units are enough per worker, so a
        # 80-MP A40 can theoretically run 4 workers.  Even though we limit
        # parallelism through an actor pool, if we do not request GPUs, Ray will
        # keep us from accessing them.
        remote = ray.remote(
            num_cpus=pc.backend_threads,
            num_gpus=20 / _cuda_props.multi_processor_count,
        )
    else:
        # if we don't have CUDA, don't request GPU
        remote = ray.remote(
            num_cpus=pc.total_threads,
            num_gpus=0,
        )

    return remote(actor)


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
        assert isinstance(ranked, RecommendationList), f"reranked has unexpected type {type(ranked)} in pipeline {name}"
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
        assert isinstance(
            reranked, RecommendationList
        ), f"reranked has unexpected type {type(reranked)} in pipeline {name}"
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
        assert isinstance(embedded, CandidateSet), f"embedded has unexpected type {type(embedded)} in pipeline {name}"
        assert hasattr(embedded, "embeddings")

        for idx, article in enumerate(embedded.articles):
            embeddings[article.article_id] = embedded.embeddings[idx].cpu().numpy()  # type: ignore
    return output_df, embeddings


def generate_profile_recs(dataset: str, outs: RecOutputs, n_profiles: int | None = None):
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
        pc = get_parallel_config()
        if pc.processes > 1:
            logger.info("starting evaluation with %d workers", pc.processes)
            init_cluster(global_logging=True)

            emb_out = ray.remote(EmbeddingWriter).remote(outs)

            gen_actor = dynamic_remote(RecGenerator)
            actors = [gen_actor.remote(outs, emb_out) for _i in range(pc.processes)]
            pool = ray.util.ActorPool(actors)
            batches = it.batched(profile_iter, BATCH_SIZE)

            for rbatch in pool.map_unordered(lambda a, b: a.generate_batch.remote(b), batches):
                pb.update(len(rbatch))

            logger.info("closing actors")
            rusage = [ray.get(actor.finish.remote()) for actor in actors]
            ray.get(emb_out.close.remote())

        else:
            logger.info("starting serial evaluation")
            # directly call things in-process
            emb_out = EmbeddingWriter(outs)
            gen = RecGenerator(outs, emb_out)

            for request in profile_iter:
                gen.generate(request)
                pb.update()

            gen.finish()
            emb_out.close()
            rusage = None

    timer.stop()
    logger.info("finished recommending in %s", timer)
    return rusage
