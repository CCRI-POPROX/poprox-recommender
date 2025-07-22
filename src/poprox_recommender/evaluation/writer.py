"""
Simple batched incremental writing for evaluation output files.
"""

# pyright: basic
import logging
from os import fspath
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, Unpack

import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetWriter

logger = logging.getLogger(__name__)

# 1M row batches by default
TARGET_BATCH_ROWS = 1024 * 1024


class ParquetWriterOptions(TypedDict):
    """
    Options for parquet writers.
    """

    compression: NotRequired[Literal["gzip", "zstd", "snappy"]]
    compression_level: NotRequired[int]


class ParquetBatchedWriter:
    """
    Incrementally rite tables to a Parquet file, with internal batching to
    reduce writing overhead.  Each table must have the same schema.
    """

    path: Path
    writer: ParquetWriter | None = None
    _writer_args: ParquetWriterOptions
    _batch: list[pa.Table]
    _batch_rows: int

    def __init__(self, path: Path, **args: Unpack[ParquetWriterOptions]):
        self.path = path

        self._batch = []
        self._batch_rows = 0
        self._writer_args = args
        if not self._writer_args:
            # default zstd compression
            self._writer_args = {"compression": "zstd", "compression_level": 3}

    def write_frame(self, df: pd.DataFrame | pa.Table):
        if isinstance(df, pd.DataFrame):
            df = pa.Table.from_pandas(df)
        if df.num_rows == 0:
            logger.debug("%s: skipping empty frame", self.path)
            return

        if self.writer is None:
            logger.info("opening output file %s", self.path)
            self.writer = ParquetWriter(fspath(self.path), df.schema, **self._writer_args)

        logger.debug("%s: adding frame of %d rows", self.path, len(df))

        self._batch.append(df)
        self._batch_rows += df.num_rows

        self._write_batch(force=False)

    def _write_batch(self, force: bool):
        assert self.writer is not None

        if self._batch_rows == 0 or (self._batch_rows < TARGET_BATCH_ROWS and not force):
            return

        logger.debug("%s: writing batch of %d rows", self.path, self._batch_rows)
        all_tbl = pa.concat_tables(self._batch, promote_options="default")
        self.writer.write(all_tbl)
        self._batch = []
        self._batch_rows = 0

    def close(self):
        if self.writer is not None:
            logger.info("closing output file %s", self.path)
            self._write_batch(force=True)
            self.writer.close()
            self.writer = None
