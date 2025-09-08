import copyreg
import itertools as it
from collections.abc import Iterator

import ray
import torch
from humanize import naturaldelta
from lenskit.logging import Progress, Task, get_logger, item_progress
from lenskit.parallel.config import ParallelConfig, get_parallel_config, subprocess_config
from lenskit.parallel.ray import TaskLimiter, init_cluster
from lenskit.pipeline import PipelineState

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import CandidateSet
from poprox_recommender.config import default_device
from poprox_recommender.data.mind import TEST_REC_COUNT, MindData
from poprox_recommender.evaluation.generate.outputs import (
    EmbeddingWriter,
    JSONRecommendationWriter,
    ParquetRecommendationWriter,
    RecommendationWriter,
    RecOutputs,
)
from poprox_recommender.recommenders.load import get_pipeline
from poprox_recommender.rusage import pretty_time

logger = get_logger(__name__)

BATCH_SIZE = 25
STAGES = ["final", "ranked", "reranked"]
# outputs we want for the result, to pre-filter
TO_SAVE = ["candidate-embedder", "recommender", "ranker", "reranker"]


def generate_profile_recs(dataset: MindData, outs: RecOutputs, pipeline: str, n_profiles: int | None = None):
    logger.info("generating recommendations", dataset=dataset.name)

    profile_iter = dataset.iter_profiles()
    if n_profiles is None:
        n_profiles = dataset.n_profiles
        logger.info("recommending for all %d profiles", n_profiles)
    else:
        logger.info("running on subset of %d profiles", n_profiles)
        profile_iter = it.islice(profile_iter, n_profiles)

    with (
        item_progress("recommend", total=n_profiles) as pb,
        Task(
            f"generate-{dataset.name}-{pipeline}",
            tags=["poprox", "generate", dataset.name, pipeline],
        ) as task,
    ):
        task.save_to_file(outs.base_dir / "generate-task.json")
        pc = get_parallel_config()
        if pc.processes > 1:
            cluster_recommend(pipeline, profile_iter, outs, pc, task, pb)

        else:
            serial_recommend(pipeline, profile_iter, outs, pb)

    logger.info("finished recommending in %s", naturaldelta(task.duration) if task.duration else "unknown time")
    cpu = task.total_cpu()
    if cpu:
        logger.info("recommendation took %s CPU", pretty_time(cpu))


def serial_recommend(pipeline: str, profiles: Iterator[RecommendationRequest], outs: RecOutputs, pb: Progress):
    logger.info("starting serial evaluation")
    # directly call things in-process
    writers: list[RecommendationWriter] = [
        ParquetRecommendationWriter(outs),
        JSONRecommendationWriter(outs),
        EmbeddingWriter(outs),
    ]

    for request in profiles:
        state = recommend_for_profile(pipeline, request)
        for w in writers:
            w.write_recommendations(request, state)
        pb.update()

    for w in writers:
        w.close()


def cluster_recommend(
    pipeline: str,
    profiles: Iterator[RecommendationRequest],
    outs: RecOutputs,
    pc: ParallelConfig,
    task: Task,
    pb: Progress,
):
    logger.info("starting parallel evaluation with task limit of %d", pc.processes)
    init_cluster(global_logging=True)

    writers = [
        ParquetRecommendationWriter.make_actor().remote(outs),
        JSONRecommendationWriter.make_actor().remote(outs),
        EmbeddingWriter.make_actor().remote(outs),
    ]

    rec_batch = dynamic_remote(recommend_batch)
    limit = TaskLimiter(pc.processes)

    writes = []
    for n, btask, bwrites in limit.imap(
        lambda batch: rec_batch.remote(pipeline, batch, writers), it.batched(profiles, BATCH_SIZE)
    ):
        pb.update(n)
        task.add_subtask(btask)
        writes += bwrites

        # wait for pending writes
        while len(writes) >= 50:
            done, writes = ray.wait(writes)
            for rh in done:
                ray.get(rh)

    logger.debug("waiting for remaining writes")
    # clear pending writes
    while writes:
        done, writes = ray.wait(writes)
        for rh in done:
            ray.get(rh)

    logger.info("closing writers")
    close = [w.close.remote() for w in writers]
    for cr in close:
        wt = ray.get(cr)
        task.add_subtask(wt)


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

    outputs = []

    with Task("generate-batch", subprocess=True, reset_hwm=True) as task:
        for request in batch:
            state = recommend_for_profile(pipeline, request)
            state = {k: v for (k, v) in state.items() if k in TO_SAVE}
            outputs.append((request, state))

        outputs = ray.put(outputs)
        writes = [w.write_recommendation_batch.remote(outputs) for w in writers]

    return len(batch), task, writes


def dynamic_remote(task_or_actor):
    """
    Dynamically configure the resource requirements of a task or actor based on
    CUDA availability and parallelism configuration.
    """
    pc = subprocess_config()
    logger.debug("worker parallel config: %s", pc)
    if torch.cuda.is_available():
        _cuda_props = torch.cuda.get_device_properties()
        # Let's take a wild guess that 20 MP units are enough per worker, so a
        # 80-MP A40 can theoretically run 4 workers.  If we do not request GPUs,
        # Ray will keep us from accessing them.
        gpu_frac = 20 / _cuda_props.multi_processor_count
        logger.debug("setting up GPU task with %d CPU, %.3f GPU", pc.backend_threads, gpu_frac)
        remote = ray.remote(
            num_cpus=pc.backend_threads,
            num_gpus=gpu_frac,
            # reuse worker processes between batches
            max_calls=0,
        )
    else:
        # if we don't have CUDA, don't request GPU
        logger.debug("setting up remote CPU-only task with %d threads", pc.total_threads)
        remote = ray.remote(
            num_cpus=pc.total_threads,
            num_gpus=0,
        )

    return remote(task_or_actor)


def _pickle_tensor(tensor: torch.Tensor):
    """
    Pickle support function to pickle a tensor, transferring to CPU.
    """
    return torch.from_numpy, (tensor.cpu().numpy(),)


copyreg.pickle(torch.Tensor, _pickle_tensor)
