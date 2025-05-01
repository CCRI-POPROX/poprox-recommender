import itertools as it

import ray
import torch
from lenskit.logging import get_logger, item_progress
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import init_cluster
from lenskit.pipeline import PipelineState
from lenskit.util import Stopwatch

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import CandidateSet
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT
from poprox_recommender.evaluation.generate.outputs import (
    EmbeddingWriter,
    JSONRecommendationWriter,
    ParquetRecommendationWriter,
    RecommendationWriter,
    RecOutputs,
)
from poprox_recommender.recommenders.load import get_pipeline

logger = get_logger(__name__)

BATCH_SIZE = 10
STAGES = ["final", "ranked", "reranked"]


def generate_profile_recs(dataset: str, outs: RecOutputs, pipeline: str, n_profiles: int | None = None):
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

            writers = [
                ray.remote(ParquetRecommendationWriter).remote(outs),
                ray.remote(JSONRecommendationWriter).remote(outs),
                ray.remote(EmbeddingWriter).remote(outs),
            ]

            task = dynamic_remote(recommend_batch)

            tasks = []
            for batch in it.batched(profile_iter, BATCH_SIZE):
                # backpressure
                while len(tasks) >= pc.processes:
                    logger.debug("waiting for workers to finish")
                    done, tasks = ray.wait(tasks)
                    # update # of finished items
                    pb.update(sum(ray.get(r) for r in done))

                tasks.append(task.remote(pipeline, batch, writers))

            logger.debug("waiting for remaining actors")
            while tasks:
                done, tasks = ray.wait(tasks)
                pb.update(sum(ray.get(r) for r in done))

            logger.info("closing writers")
            close = [w.close.remote() for w in writers]
            while close:
                _, close = ray.wait(close)

            logger.info("finished recommending")

        else:
            logger.info("starting serial evaluation")
            # directly call things in-process
            writers = [
                ParquetRecommendationWriter(outs),
                JSONRecommendationWriter(outs),
                EmbeddingWriter(outs),
            ]

            for request in profile_iter:
                state = recommend_for_profile(pipeline, request)
                for w in writers:
                    w.write_recommendations(request, state)
                pb.update()

            for w in writers:
                w.close()
            rusage = None

    timer.stop()
    logger.info("finished recommending in %s", timer)
    return rusage


def recommend_for_profile(pipeline: str, request: RecommendationRequest) -> PipelineState:
    """
    Generate recommendations for a single request, returning the pipeline state.
    """
    # get_pipeline caches, so this will load once per worker
    pipe = get_pipeline(pipeline, device=default_device())
    log = logger.bind(profile_id=str(request.interest_profile.profile_id))
    log.debug("beginning recommendation")
    if request.num_recs != TEST_REC_COUNT:
        log.warning(
            "request for %s had unexpected recommendation count %d",
            request.interest_profile.profile_id,
            request.num_recs,
        )

    inputs = {
        "candidate": CandidateSet(articles=request.todays_articles),
        "clicked": CandidateSet(articles=request.past_articles),
        "profile": request.interest_profile,
    }

    try:
        return pipe.run_all(**inputs)
    except Exception as e:
        logger.error("error recommending for profile %s: %s", request.interest_profile.profile_id, e)
        raise e


def recommend_batch(pipeline, batch: list[RecommendationRequest], writers: list[RecommendationWriter]):
    """
    Batch-recommend function, to be used as a Ray worker task.
    """

    writes = []

    for request in batch:
        state = recommend_for_profile(pipeline, request)
        # put once, so all actors share the object
        state = ray.put(state)
        # rate-limit our requests to the writers
        while len(writes) >= 10:
            _done, writes = ray.wait(writes)
        writes += [w.write_recommendations.remote(request, state) for w in writers]

    # wait for outstanding writes on this batch
    while writes:
        _done, writes = ray.wait(writes)

    return len(batch)


def dynamic_remote(task_or_actor):
    """
    Dynamically configure the resource requirements of a task or actor based on
    CUDA availability and parallelism configuration.
    """
    pc = get_parallel_config()
    if torch.cuda.is_available():
        _cuda_props = torch.cuda.get_device_properties()
        # Let's take a wild guess that 20 MP units are enough per worker, so a
        # 80-MP A40 can theoretically run 4 workers.  If we do not request GPUs,
        # Ray will keep us from accessing them.
        remote = ray.remote(
            num_cpus=pc.backend_threads,
            num_gpus=20 / _cuda_props.multi_processor_count,
        )
    else:
        # if we don't have CUDA, don't request GPU
        logger.debug("setting up remote CPU-only task with %d threads", pc.total_threads)
        remote = ray.remote(
            num_cpus=pc.total_threads,
            num_gpus=0,
        )

    return remote(task_or_actor)
