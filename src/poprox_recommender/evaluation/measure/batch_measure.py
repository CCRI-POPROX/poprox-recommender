# pyright: basic
import logging
from collections.abc import Sequence
from itertools import batched
from typing import Any, Iterator
from uuid import UUID

import ray
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import TaskLimiter, init_cluster

from poprox_recommender.data.eval import EvalData
from poprox_recommender.evaluation.measure.recloader import RecLoader
from poprox_recommender.evaluation.metrics import measure_rec_metrics

logger = logging.getLogger(__name__)


def recommendation_eval_results(eval_data: EvalData, rec_data: RecLoader) -> Iterator[dict[str, Any]]:
    pc = get_parallel_config()
    if pc.processes > 1:
        logger.info("starting parallel measurement with up to %d tasks", pc.processes)
        init_cluster(global_logging=True, configure_logging=False)

        eval_data_ref = ray.put(eval_data)
        rec_data_ref = ray.put(rec_data)
        limit = TaskLimiter(pc.processes)

        for bres in limit.imap(
            lambda batch: measure_batch.remote(batch, eval_data_ref, rec_data_ref),
            batched(rec_data.iter_slate_ids(), 100),
            ordered=False,
        ):
            assert isinstance(bres, list)
            yield from bres

    else:
        for slate in rec_data.iter_slate_ids():
            yield measure_slate(slate, eval_data, rec_data)


def measure_slate(slate_id: UUID, eval_data: EvalData, rec_data: RecLoader):
    recs = rec_data.slate_recs(slate_id)
    truth = eval_data.slate_truth(slate_id)
    assert truth is not None
    if len(truth) == 0:
        logger.debug("slate %s has no truth", slate_id)
    return measure_rec_metrics(slate_id, recs, truth, eval_data)


@ray.remote(num_cpus=2)
def measure_batch(slate_ids: Sequence[UUID], eval_data: EvalData, rec_data: RecLoader) -> list[dict[str, Any]]:
    """
    Measure a batch of recommendations.
    """
    return [measure_slate(slate_id, eval_data, rec_data) for slate_id in slate_ids]
