# pyright: basic
import logging
from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from typing import Any, Iterator

import ray
from lenskit.parallel import get_parallel_config
from lenskit.parallel.ray import TaskLimiter, init_cluster
from xopen import xopen

from poprox_recommender.data.eval import EvalData
from poprox_recommender.evaluation.generate.outputs import OfflineRecommendations
from poprox_recommender.evaluation.metrics import measure_rec_metrics

logger = logging.getLogger(__name__)
BATCH_SIZE = 1000


def recommendation_eval_results(eval_data: EvalData, rec_path: Path) -> Iterator[dict[str, Any]]:
    pc = get_parallel_config()
    with xopen(rec_path, "rb") as f:
        if pc.processes > 1:
            logger.info("starting parallel measurement with up to %d tasks", pc.processes)
            init_cluster(global_logging=True, configure_logging=False)
            eval_data_ref = ray.put(eval_data)
            limit = TaskLimiter(pc.processes)
            for bres in limit.imap(
                lambda batch: measure_batch.remote(batch, eval_data_ref),
                batched(f, BATCH_SIZE),
                ordered=False,
            ):
                assert isinstance(bres, list)
                yield from bres
        else:
            for batch in batched(f, BATCH_SIZE):
                for line in batch:
                    yield measure_slate(line, eval_data)


def measure_slate(raw_line: bytes, eval_data: EvalData):
    rec = OfflineRecommendations.model_validate_json(raw_line)
    truth = eval_data.slate_truth(rec.slate_id)
    assert truth is not None
    if len(truth) == 0:
        logger.debug("slate %s has no truth", rec.slate_id)
    return measure_rec_metrics(rec.slate_id, rec.results, truth, eval_data)


@ray.remote(num_cpus=2)
def measure_batch(batch: Sequence[bytes], eval_data: EvalData) -> list[dict[str, Any]]:
    """
    Measure a batch of recommendations.
    """
    return [measure_slate(line, eval_data) for line in batch]
