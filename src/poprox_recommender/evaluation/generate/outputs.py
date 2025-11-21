"""
This file provides abstractions for saving the outputs of batch-running
recommender pipelines, so the worker code just needs to know about a "save my
stuff" interface and can be spared the details of output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection
from pathlib import Path
from typing import Any, ClassVar, Generic, TextIO
from uuid import UUID

import numpy as np
import pandas as pd
import pyarrow as pa
import ray
import torch
import zstandard
from lenskit.logging import Task, get_logger
from lenskit.pipeline import PipelineState
from pydantic import BaseModel
from typing_extensions import TypeVar

from poprox_concepts.api.recommendations import RecommendationRequest, RecommendationRequestV4
from poprox_concepts.domain import CandidateSet, ImpressedSection
from poprox_recommender.evaluation.writer import ParquetBatchedWriter

logger = get_logger(__name__)

Package = TypeVar("Package", default=Any)


class OfflineRecommendations(BaseModel):
    slate_id: UUID
    request: RecommendationRequestV4
    results: OfflineRecResults


class OfflineRecResults(BaseModel, validate_assignment=True):
    final: ImpressedSection
    ranked: ImpressedSection | None = None
    reranked: ImpressedSection | None = None


class RecOutputs:
    """
    Abstraction of output layout to put it in one place.
    """

    base_dir: Path

    def __init__(self, dir: Path):
        self.base_dir = dir

    @property
    def rec_dir(self):
        """
        Output directory for recommendations.  This entire directory should be read
        with :func:`pd.read_parquet`, and it will load the shareds from it.
        """
        return self.base_dir / "recommendations"

    @property
    def rec_parquet_file(self):
        """
        Output file for recommendations in Parquet tabular format.
        """
        return self.base_dir / "recommendations.parquet"

    @property
    def rec_json_file(self):
        """
        Output file for recommendations in NDJSON format.
        """
        return self.base_dir / "recommendations.ndjson.zst"

    @property
    def emb_file(self):
        """
        File for final deduplicated embedding outputs.
        """
        return self.base_dir / "embeddings.parquet"


class RecommendationWriter(ABC, Generic[Package]):
    """
    Interface for recommendation writers that write various aspects of
    recommendations to disk.

    This class is a little weird, because it supports both local and Ray actor
    writing, through an abstraction of a "package" that gets prepared and then
    written.  This because serializing and deserializing the entire request and
    pipeline state, only to write a part of it, is the bottleneck in concurrent
    evaluation.  Therefore, a writer defines two methods:

    - :meth:`prepare_write` takes the request and pipeline outputs and prepares
      a “write package”, which can be any pickleable object.  When the writer is
      instantiated as an actor, this method will be called in the worker
      process, and its return value sent to the actor.
    - :meth:`write_package` takes a write package returned from
      :meth:`prepare_write` and actually writes it to the output.

    The base class (this class) defines :meth:`write_recommendations` and
    :meth:`write_recommendation_batch` that use these methods to more
    efficiently write recommendation results to disk.

    Subclasses must be instantiable with a no-argument constructor, resulting in
    a writer whose :meth:`write_package` method may fail.
    """

    WANTED_NODES: ClassVar[Collection[str]] = {}

    task: Task

    def __init__(self, *, backend: ray.ObjectRef | None = None):
        name = self.__class__.__name__
        self.task = Task(f"write-{name}", tags=["output", name], subprocess=True)
        self.task.start()

    @classmethod
    def create_remote(cls, *args, **kwargs) -> RecommendationWriter[Package]:
        """
        Instantiates this recommendation writer with a remote writing backend.
        Takes the same arguments as the class constructor.

        Returns:
            A proxy writer that uses the remote backend writer to write.
        """
        acls = ray.remote(num_cpus=1)(cls)
        remote = acls.remote(*args, **kwargs)
        local = cls()
        return ProxyRecommendationWriter(local=local, remote=remote)

    @abstractmethod
    def prepare_write(self, slate_id: UUID, request: RecommendationRequest, pipeline_state: PipelineState) -> Package:
        """
        Create a package representing the data needed to write.
        """

    @abstractmethod
    def write_package(self, package: Package):
        """
        Write a data package to storage.
        """

    def write_package_batch(self, packages: list[Package]):
        """
        Write a batch of packages, to decrease trannsmission overhead.
        """
        for package in packages:
            self.write_package(package)

    def write_recommendations(self, slate_id: UUID, request: RecommendationRequest, pipeline_state: PipelineState):
        """
        Write recommendations to this writer's storage.
        """
        pkg = self.prepare_write(slate_id, request, pipeline_state)
        return self.write_package(pkg)

    def close(self):
        """
        Close the writer, returning a task capturing its work.  Subclasses must
        call the base class implementation and return its return value.
        """
        self.task.finish()
        return self.task

    def write_recommendation_batch(self, batch: list[tuple[UUID, RecommendationRequest, PipelineState]]):
        """
        Write a batch of recommendations. In remote mode, this will return the Ray task
        that can be waited for with :func:`ray.get` or :func:`ray.wait`.
        """
        packages = [self.prepare_write(rid, req, state) for rid, req, state in batch]
        return self.write_package_batch(packages)


class ProxyRecommendationWriter(RecommendationWriter[Package]):
    """
    A proxy object that writes with an underlying backend.
    """

    _local: RecommendationWriter[Package]
    _remote: ray.ObjectRef[RecommendationWriter]

    def __init__(self, *, local: RecommendationWriter[Package], remote: ray.ObjectRef[Any]):
        self._local = local
        self._remote = remote

    def prepare_write(self, slate_id: UUID, request: RecommendationRequest, pipeline_state: PipelineState) -> Package:
        return self._local.prepare_write(slate_id, request, pipeline_state)

    def write_package(self, package: Package):
        return self._remote.write_package.remote(package)  # type: ignore

    def write_package_batch(self, packages: list[Package]):
        return self._remote.write_package_batch.remote(packages)  # pyright: ignore[reportAttributeAccessIssue]

    def close(self):
        self._local.close()
        task = ray.get(self._remote.close.remote())  # type: ignore
        # skip super, we use remote task
        return task


class ParquetRecommendationWriter(RecommendationWriter[list[pd.DataFrame]]):
    """
    Implementation of :class:`RecommendationWriter` that writes the recommendations in
    tabular format to Parquet for easy analysis.

    Can be used as a Ray actor.
    """

    WANTED_NODES = {"recommender", "ranker", "reranker"}

    path: Path
    writer: ParquetBatchedWriter

    def __init__(self, outs: RecOutputs | None = None):
        super().__init__()
        if outs is not None:
            self.path = outs.rec_parquet_file
            outs.rec_parquet_file.parent.mkdir(exist_ok=True, parents=True)
            self.writer = ParquetBatchedWriter(outs.rec_parquet_file, compression="snappy")

    def prepare_write(
        self, slate_id: UUID, request: RecommendationRequest, pipeline_state: PipelineState
    ) -> list[pd.DataFrame]:
        logger.debug("writing recommendations to Parquet", slate_id=slate_id)
        # recommendations {account id (uuid): LIST[Article]}
        # use the url of Article
        # profile = request.interest_profile.profile_id

        # get the different recommendation lists to record
        recs = pipeline_state["recommender"]
        frames = [
            pd.DataFrame(
                {
                    "slate_id": str(slate_id),
                    "stage": "final",
                    "item_id": [str(impression.article.article_id) for impression in recs.impressions],
                    "rank": np.arange(len(recs.impressions), dtype=np.int16) + 1,
                }
            )
        ]
        ranked = pipeline_state.get("ranker", None)
        if ranked is not None:
            assert isinstance(ranked, ImpressedSection), f"reranked has unexpected type {type(ranked)}"
            frames.append(
                pd.DataFrame(
                    {
                        "slate_id": str(slate_id),
                        "stage": "ranked",
                        "item_id": [str(impression.article.article_id) for impression in ranked.impressions],
                        "rank": np.arange(len(ranked.impressions), dtype=np.int16) + 1,
                    }
                )
            )
        reranked = pipeline_state.get("reranker", None)
        if reranked is not None:
            assert isinstance(reranked, ImpressedSection), f"reranked has unexpected type {type(reranked)}"
            frames.append(
                pd.DataFrame(
                    {
                        "slate_id": str(slate_id),
                        "stage": "reranked",
                        "item_id": [str(impression.article.article_id) for impression in reranked.impressions],
                        "rank": np.arange(len(reranked.impressions), dtype=np.int16) + 1,
                    }
                )
            )

        return frames

    def write_package(self, package: list[pd.DataFrame]):
        for df in package:
            self.writer.write_frame(df)

    def close(self):
        if hasattr(self, "writer"):
            self.writer.close()
        return super().close()


class JSONRecommendationWriter(RecommendationWriter[str]):
    """
    Implementation of :class:`RecommendationWriter` that writes the recommendations in
    compressed NDJSON format for detailed analysis.

    Can be used as a Ray actor.
    """

    WANTED_NODES = {"recommender", "ranker", "reranker"}

    writer: TextIO

    def __init__(self, outs: RecOutputs | None = None):
        super().__init__()
        if outs is not None:
            outs.rec_parquet_file.parent.mkdir(exist_ok=True, parents=True)
            self.writer = zstandard.open(outs.rec_json_file, "wt", zstandard.ZstdCompressor(1))

    def prepare_write(self, slate_id: UUID, request: RecommendationRequest, pipeline_state: PipelineState) -> str:
        logger.debug("writing recommendations to JSON", slate_id=slate_id)
        # recommendations {account id (uuid): LIST[Article]}
        # use the url of Article

        # get the different recommendation lists to record

        recs = pipeline_state["recommender"]
        results = OfflineRecResults(final=recs)

        ranked = pipeline_state.get("ranker", None)
        if ranked is not None:
            results.ranked = ranked

        reranked = pipeline_state.get("reranker", None)
        if reranked is not None:
            results.reranked = reranked

        data = OfflineRecommendations(slate_id=slate_id, request=request, results=results)
        return data.model_dump_json(serialize_as_any=True, fallback=_json_fallback)

    def write_package(self, package: str):
        print(package, file=self.writer)

    def close(self):
        if hasattr(self, "writer"):
            self.writer.close()
        return super().close()


def _json_fallback(v):
    if isinstance(v, torch.Tensor):
        return v.tolist()
    else:
        return v


class EmbeddingWriter(RecommendationWriter[pa.Table | None]):
    """
    Implementation of :class:`RecommendationWriter` that extracts the candidate embeddings and writes them to disk.

    Can be used as a Ray actor.
    """

    WANTED_NODES = {"candidate-selector"}

    outputs: RecOutputs
    pkg_seen: set[str]
    write_seen: set[str]
    writer: ParquetBatchedWriter

    def __init__(self, outs: RecOutputs | None = None):
        super().__init__()
        self.pkg_seen = set()
        self.write_seen = set()
        if outs is not None:
            self.outputs = outs
            outs.rec_parquet_file.parent.mkdir(exist_ok=True, parents=True)
            self.writer = ParquetBatchedWriter(self.outputs.emb_file, compression="snappy")

    def prepare_write(
        self, slate_id: UUID, request: RecommendationRequest, pipeline_state: PipelineState
    ) -> pa.Table | None:
        # get the embeddings
        embedded = pipeline_state.get("candidate-embedder", None)
        rows = []
        if embedded is not None:
            assert isinstance(embedded, CandidateSet), f"embedded has unexpected type {type(embedded)}"
            assert hasattr(embedded, "embeddings")

            for idx, article in enumerate(embedded.articles):
                aid = str(article.article_id)
                # first-stage filtering, so we only send embeddings once from each worker
                if aid not in self.pkg_seen:
                    rows.append({"article_id": aid, "embedding": embedded.embeddings[idx].cpu().numpy()})  # type: ignore
                    self.pkg_seen.add(aid)

        if rows:
            # directly use pyarrow to avoid DF overhead, small but easy to avoid here
            logger.debug("sending %d embedding rows", len(rows))
            emb_tbl = pa.Table.from_pylist(rows)
            return emb_tbl

    def write_package(self, package: pa.Table | None):
        if package is None:
            return

        # second-stage filtering, so we only write embeddings once
        # this is redundant in single-process eval, necessary in multi-process
        article_ids = package.column("article_id").to_pylist()
        mask = [aid not in self.write_seen for aid in article_ids]
        mask = pa.array(mask, pa.bool_())

        package = package.filter(mask)
        if package.num_rows:
            logger.debug("writing %d embeddings rows", package.num_rows)
            self.writer.write_frame(package)

        for aid in article_ids:
            self.write_seen.add(aid)  # type: ignore

    def close(self):
        if hasattr(self, "writer"):
            self.writer.close()
        return super().close()
