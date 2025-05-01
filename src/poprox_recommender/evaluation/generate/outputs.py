from __future__ import annotations

from pathlib import Path
from typing import Protocol, TextIO

import numpy as np
import pandas as pd
import pyarrow as pa
import zstandard
from lenskit.pipeline import PipelineState
from pydantic import BaseModel

from poprox_concepts.api.recommendations import RecommendationRequest
from poprox_concepts.domain import CandidateSet, RecommendationList
from poprox_recommender.evaluation.writer import ParquetBatchedWriter


class OfflineRecommendations(BaseModel):
    request: RecommendationRequest
    results: OfflineRecResults


class OfflineRecResults(BaseModel):
    final: RecommendationList
    topn: RecommendationList | None = None
    reranked: RecommendationList | None = None


class RecOutputs:
    """
    Abstraction of output layout to put it in one place.
    """

    base_dir: Path

    rec_writer: ParquetBatchedWriter

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

    def open(self, part: int | str | None):
        """
        Open the writers for collecting recommendation output.
        """
        if part is None:
            fn = "data.parquet"
        else:
            fn = f"part-{part}.parquet"

        self.rec_dir.mkdir(exist_ok=True, parents=True)
        self.rec_writer = ParquetBatchedWriter(self.rec_dir / fn)

    def close(self):
        self.rec_writer.close()

    def __getstate__(self):
        # only the base dir is pickled
        return {"base_dir": self.base_dir}


class RecommendationWriter(Protocol):
    """
    Interface for recommendation writers that write various aspects of recommendations to disk.
    """

    def write_recommendations(self, request: RecommendationRequest, pipeline_state: PipelineState):
        """
        Write recommendations to this writer's storage.
        """
        ...


class ParquetRecommendationWriter:
    """
    Implementation of :class:`RecommendationWriter` that writes the recommendations in
    tabular format to Parquet for easy analysis.

    Can be used as a Ray actor.
    """

    writer: ParquetBatchedWriter

    def __init__(self, outs: RecOutputs):
        self.writer = ParquetBatchedWriter(outs.rec_parquet_file)

    def write_recommendations(self, request: RecommendationRequest, pipeline_state: PipelineState):
        assert isinstance(request, RecommendationRequest)
        # recommendations {account id (uuid): LIST[Article]}
        # use the url of Article
        profile = request.interest_profile.profile_id
        assert profile is not None

        # get the different recommendation lists to record
        recs = pipeline_state["recommender"]
        self.writer.write_frame(
            pd.DataFrame(
                {
                    "profile_id": str(profile),
                    "stage": "final",
                    "item_id": [str(a.article_id) for a in recs.articles],
                    "rank": np.arange(len(recs.articles), dtype=np.int16) + 1,
                }
            )
        )
        ranked = pipeline_state.get("ranker", None)
        if ranked is not None:
            assert isinstance(ranked, RecommendationList), f"reranked has unexpected type {type(ranked)}"
            self.writer.write_frame(
                pd.DataFrame(
                    {
                        "profile_id": str(profile),
                        "stage": "ranked",
                        "item_id": [str(a.article_id) for a in ranked.articles],
                        "rank": np.arange(len(ranked.articles), dtype=np.int16) + 1,
                    }
                )
            )
        reranked = pipeline_state.get("reranker", None)
        if reranked is not None:
            assert isinstance(reranked, RecommendationList), f"reranked has unexpected type {type(reranked)}"
            self.writer.write_frame(
                pd.DataFrame(
                    {
                        "profile_id": str(profile),
                        "stage": "reranked",
                        "item_id": [str(a.article_id) for a in reranked.articles],
                        "rank": np.arange(len(reranked.articles), dtype=np.int16) + 1,
                    }
                )
            )

    def close(self):
        self.writer.close()


class JSONRecommendationWriter:
    """
    Implementation of :class:`RecommendationWriter` that writes the recommendations in
    compressed NDJSON format for detailed analysis.

    Can be used as a Ray actor.
    """

    writer: TextIO

    def __init__(self, outs: RecOutputs):
        self.writer = zstandard.open(outs.rec_json_file, "wt", zstandard.ZstdCompressor(6))

    def write_recommendations(self, request: RecommendationRequest, pipeline_state: PipelineState):
        assert isinstance(request, RecommendationRequest)
        # recommendations {account id (uuid): LIST[Article]}
        # use the url of Article
        profile = request.interest_profile.profile_id
        assert profile is not None

        # get the different recommendation lists to record
        recs = pipeline_state["recommender"]
        results = OfflineRecResults(final=recs)

        ranked = pipeline_state.get("ranker", None)
        if ranked is not None:
            results.topn = ranked

        reranked = pipeline_state.get("reranker", None)
        if reranked is not None:
            results.reranked = reranked

        data = OfflineRecommendations(request=request, results=results)
        print(data.model_dump_json(), file=self.writer)

    def close(self):
        self.writer.close()


class EmbeddingWriter:
    """
    Implementation of :class:`RecommendationWriter` that extracts the candidate embeddings and writes them to disk.

    Can be used as a Ray actor.
    """

    outputs: RecOutputs
    seen: set[str]
    writer: ParquetBatchedWriter

    def __init__(self, outs: RecOutputs):
        self.outputs = outs
        self.seen = set()
        self.writer = ParquetBatchedWriter(self.outputs.emb_file, compression="snappy")

    def write_recommendations(self, request: RecommendationRequest, pipeline_state: PipelineState):
        # get the embeddings
        embedded = pipeline_state.get("candidate-embedder", None)
        rows = []
        if embedded is not None:
            assert isinstance(embedded, CandidateSet), f"embedded has unexpected type {type(embedded)}"
            assert hasattr(embedded, "embeddings")

            for idx, article in enumerate(embedded.articles):
                aid = str(article.article_id)
                if aid not in self.seen:
                    rows.append({"article_id": aid, "embedding": embedded.embeddings[idx].cpu().numpy()})  # type: ignore

        if rows:
            # directly use pyarrow to avoid DF overhead, small but easy to avoid here
            emb_tbl = pa.Table.from_pylist(rows)
            self.writer.write_frame(emb_tbl)

        # record the article IDs we have written for future dedup
        for aid, _e in rows:
            self.seen.add(aid)

    def close(self):
        self.writer.close()
