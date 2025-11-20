import itertools as it
from uuid import UUID

import ray
import torch
from humanize import metric, naturaldelta
from lenskit.logging import Progress, Task, get_logger, item_progress
from lenskit.parallel.config import ParallelConfig, get_parallel_config, subprocess_config
from lenskit.parallel.ray import TaskLimiter, init_cluster
from lenskit.pipeline import Pipeline, PipelineState
from torch.multiprocessing.reductions import reduce_tensor

from poprox_concepts.api.recommendations import RecommendationRequestV4
from poprox_recommender.config import default_device
from poprox_recommender.data.eval import EvalData
from poprox_recommender.data.mind import TEST_REC_COUNT
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


def generate_recs_for_requests(dataset: EvalData, outs: RecOutputs, pipeline: str, n_requests: int | None = None):
    logger.info("generating recommendations", dataset=dataset.name)

    pc = get_parallel_config()
    cluster = pc.processes > 1

    count = n_requests if n_requests is not None else dataset.n_requests
    logger.info("recommending for %d requests", count)

    with (
        item_progress("recommend", total=count) as pb,
        Task(
            f"generate-{dataset.name}-{pipeline}",
            tags=["poprox", "generate", dataset.name, pipeline],
        ) as task,
    ):
        task.save_to_file(outs.base_dir / "generate-task.json")
        pc = get_parallel_config()
        if cluster:
            cluster_recommend(dataset, pipeline, n_requests, outs, pc, task, pb)
        else:
            serial_recommend(dataset, pipeline, n_requests, outs, pb)

    logger.info("finished recommending in %s", naturaldelta(task.duration) if task.duration else "unknown time")
    cpu = task.total_cpu()
    if cpu:
        logger.info("recommendation took %s CPU", pretty_time(cpu))
    if task.system_power:
        logger.info("recommendation took %s", metric(task.system_power, "J"))


def serial_recommend(dataset: EvalData, pipeline: str, n_requests: int | None, outs: RecOutputs, pb: Progress):
    logger.info("loading pipeline")
    pipe = get_pipeline(pipeline, device=default_device())
    logger.info("starting serial evaluation")
    # directly call things in-process
    writers: list[RecommendationWriter] = [
        ParquetRecommendationWriter(outs),
        JSONRecommendationWriter(outs),
        EmbeddingWriter(outs),
    ]

    for slate_id in dataset.iter_slate_ids(limit=n_requests):
        request = dataset.lookup_request(slate_id)
        state = recommend_for_request(pipe, request)
        for w in writers:
            w.write_recommendations(request, state)
        pb.update()

    for w in writers:
        w.close()


def cluster_recommend(
    dataset: EvalData,
    pipeline: str,
    max_recommendations: int | None,
    outs: RecOutputs,
    pc: ParallelConfig,
    task: Task,
    pb: Progress,
):
    logger.info("starting parallel evaluation with task limit of %d", pc.processes)
    init_cluster(global_logging=True)

    device = default_device()
    if torch.device(device).type != "cpu":
        logger.debug("registering custom serializer for shared tensors")
        ray.util.register_serializer(torch.Tensor, serializer=_SharedTensor, deserializer=torch.as_tensor)

    logger.info("loading pipeline")
    pipe = get_pipeline(pipeline, device=default_device())
    logger.info("sending pipeline to Ray cluster")
    pipe = ray.put(pipe)

    writers: list[RecommendationWriter] = [
        ParquetRecommendationWriter.create_remote(outs),
        JSONRecommendationWriter.create_remote(outs),
        EmbeddingWriter.create_remote(outs),
    ]

    slate_ids = dataset.iter_slate_ids(limit=max_recommendations)
    ds_ref = ray.put(dataset)
    rec_batch = dynamic_remote(recommend_batch)
    limit = TaskLimiter(pc.processes)

    writes = []
    for n, btask, bwrites in limit.imap(
        lambda batch: rec_batch.remote(pipe, batch, writers, dataset=ds_ref),
        it.batched(slate_ids, BATCH_SIZE),
        ordered=False,
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
    for w in writers:
        wt = w.close()
        task.add_subtask(wt)


def recommend_for_request(pipeline: Pipeline, request: RecommendationRequestV4) -> PipelineState:
    """
    Generate recommendations for a single request, returning the pipeline state.
    """
    # get_pipeline caches, so this will load once per worker
    log = logger.bind(profile_id=str(request.interest_profile.profile_id))
    log.debug("beginning recommendation")
    if request.num_recs != TEST_REC_COUNT:
        log.warning(
            "request for %s had unexpected recommendation count %d",
            request.interest_profile.profile_id,
            request.num_recs,
        )

    inputs = {
        "candidate": request.candidates,
        "clicked": request.interacted,
        "profile": request.interest_profile,
    }

    try:
        return pipeline.run_all(**inputs)
    except Exception as e:
        logger.error("error recommending for profile %s: %s", request.interest_profile.profile_id, e)
        raise e


def recommend_batch(
    pipeline: Pipeline,
    batch: list[UUID | int],
    writers: list[RecommendationWriter],
    dataset: EvalData,
):
    """
    Batch-recommend function, to be used as a Ray worker task.
    """

    outputs = []

    with Task("generate-batch", subprocess=True, reset_hwm=True) as task:
        for request_id in batch:
            request = dataset.lookup_request(request_id)
            state = recommend_for_request(pipeline, request)
            state = {k: v for (k, v) in state.items() if k in TO_SAVE}
            outputs.append((request, state))

        writes = [w.write_recommendation_batch(outputs) for w in writers]

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
        # 80-MP A40 can theoretically run 8 workers.  If we do not request GPUs,
        # Ray will keep us from accessing them.
        #
        # We also hard-code 2 CPUs per worker, because CUDA-powered eval doesn't
        # use any other parallelism at this time.
        gpu_frac = 20 / _cuda_props.multi_processor_count
        logger.debug("setting up GPU task with %d CPU, %.3f GPU", 2, gpu_frac)
        remote = ray.remote(
            num_cpus=2,
            num_gpus=gpu_frac,
            # reuse worker processes between batches
            max_calls=0,
        )
    else:
        # if we don't have CUDA, don't request GPU, and we'll need CPU threads
        logger.debug("setting up remote CPU-only task with %d threads", pc.total_threads)
        remote = ray.remote(
            num_cpus=pc.total_threads,
            num_gpus=0,
        )

    return remote(task_or_actor)


class _SharedTensor:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __reduce__(self):
        return reduce_tensor(self.tensor)
