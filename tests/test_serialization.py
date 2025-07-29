"""
Tests to benchmark and measure serialization methods.

Based on Karl's WIP at https://github.com/CCRI-POPROX/poprox-platform/pull/454.
"""

import gzip
from pathlib import Path

import numpy as np
import orjson
import zstandard
from humanize import naturalsize
from pytest import fixture, mark
from rich import box
from rich.console import Console
from rich.table import Table

from poprox_concepts.api.recommendations import RecommendationRequestV2

data_dir = Path(__file__).resolve().parent / "request_data"
req_file = data_dir / "request_body.json"


@fixture
def basic_request():
    text = req_file.read_text()
    yield RecommendationRequestV2.model_validate_json(text)


@fixture
def enhanced_request(basic_request: RecommendationRequestV2):
    req_body = basic_request.model_dump()
    embeddings = {}

    # Article embeddings
    for article in basic_request.candidates.articles:
        embeddings[article.article_id] = {
            "headline": np.random.randn(768),
            "subhead": np.random.randn(768),
            "body": np.random.randn(768),
        }

    # Image embeddings
    for article in basic_request.candidates.articles:
        for image in article.images or []:
            embeddings[image.image_id] = np.random.randn(1024)

    req_body["embeddings"] = embeddings

    yield req_body


@mark.benchmark(group="basic")
def test_pydantic_basic(basic_request, benchmark):
    def serialize():
        _serialized_req = basic_request.model_dump_json()

    benchmark(serialize)


@mark.benchmark(group="basic")
def test_orjson_basic(basic_request, benchmark):
    def serialize():
        br = basic_request.model_dump()
        _serialized_req = orjson.dumps(br, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)

    benchmark(serialize)


@mark.benchmark(group="embeddings")
def test_orjson(enhanced_request, benchmark):
    def serialize():
        _serialized_req = orjson.dumps(enhanced_request, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)

    benchmark(serialize)


@mark.benchmark(group="embeddings")
def test_orjson_gzip(enhanced_request, benchmark):
    def serialize():
        serialized_req = orjson.dumps(enhanced_request, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)
        _gizpped_req = gzip.compress(serialized_req)

    benchmark(serialize)


@mark.benchmark(group="embeddings")
def test_orjson_zstd(enhanced_request, benchmark):
    def serialize():
        serialized_req = orjson.dumps(enhanced_request, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)
        _zst_req = zstandard.compress(serialized_req, level=1)

    benchmark(serialize)


def dump_size_versions():
    console = Console()
    br = next(basic_request._fixture_function())
    er = next(enhanced_request._fixture_function(br))

    br_json = br.model_dump_json()
    er_json = orjson.dumps(er, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS)
    er_gzip = gzip.compress(er_json)
    er_zstd = zstandard.compress(er_json, level=1)

    table = Table("Method", "Size", "Increase", title="Serialization Sizes", box=box.ASCII2)

    table.add_row("Baseline", naturalsize(len(br_json)))
    table.add_section()

    table.add_row("orjson", naturalsize(len(er_json)), "{:.2%}".format(len(er_json) / len(br_json) - 1))
    table.add_row("orjson + gz", naturalsize(len(er_gzip)), "{:.2%}".format(len(er_gzip) / len(br_json) - 1))
    table.add_row("orjson + zst", naturalsize(len(er_zstd)), "{:.2%}".format(len(er_zstd) / len(br_json) - 1))

    console.print(table)


if __name__ == "__main__":
    dump_size_versions()
