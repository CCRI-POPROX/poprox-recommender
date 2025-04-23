from pathlib import Path
from uuid import UUID

import numpy as np
import pyarrow as pa

from poprox_recommender.evaluation.writer import ParquetBatchedWriter


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


class EmbeddingWriter:
    """
    Write embeddings to disk.

    Can be used as a Ray actor.
    """

    outputs: RecOutputs
    seen: set[UUID]
    writer: ParquetBatchedWriter

    def __init__(self, outs: RecOutputs):
        self.outputs = outs
        self.seen = set()
        self.writer = ParquetBatchedWriter(self.outputs.emb_file, compression="snappy")

    def write_embeddings(self, embeddings: dict[UUID, np.ndarray]):
        rows = [{"article_id": str(aid), "embedding": emb} for (aid, emb) in embeddings.items() if aid not in self.seen]
        self.seen |= embeddings.keys()
        if rows:
            # directly use pyarrow to avoid DF overhead, small but easy to avoid here
            emb_tbl = pa.Table.from_pylist(rows)
            self.writer.write_frame(emb_tbl)

    def close(self):
        self.writer.close()
