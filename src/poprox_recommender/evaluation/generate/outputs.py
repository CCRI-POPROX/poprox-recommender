from pathlib import Path

from poprox_recommender.evaluation.writer import ParquetBatchedWriter


class RecOutputs:
    """
    Abstraction of output layout to put it in one place.
    """

    base_dir: Path

    rec_writer: ParquetBatchedWriter
    emb_writer: ParquetBatchedWriter

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
    def emb_temp_dir(self):
        """
        Temporary directory for embeddings.
        """
        return self.base_dir / "embeddings.tmp"

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
        # since this is temp storage be fast
        self.emb_temp_dir.mkdir(exist_ok=True, parents=True)
        self.emb_writer = ParquetBatchedWriter(self.emb_temp_dir / fn, compression="snappy")

    def close(self):
        self.rec_writer.close()
        self.emb_writer.close()

    def __getstate__(self):
        # only the base dir is pickled
        return {"base_dir": self.base_dir}
