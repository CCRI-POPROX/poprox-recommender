import logging
from itertools import islice

from pytest import fixture, skip

from poprox_recommender.data.eval import EvalData
from poprox_recommender.testing import mind_data

logger = logging.getLogger(__name__)


def test_load(mind_data: EvalData):
    "Test the data loads & basic properties work."
    assert mind_data.n_articles > 100
    assert mind_data.n_requests > 500


def test_scan_slate_ids(mind_data: EvalData):
    "Test that the slate ID iterator is consistent with request count."
    # count the number of items in the iterator
    count = sum(1 for _ in mind_data.iter_slate_ids())
    assert count == mind_data.n_requests


def test_lookup_requests(mind_data: EvalData):
    "Test that we can look up some requests."

    for slate_id in islice(mind_data.iter_slate_ids(), 10):
        req = mind_data.lookup_request(slate_id)
        assert req is not None
        # some basic sanity checks for the current setup
        assert 5 <= len(req.candidates.articles) <= 500
        assert 0 <= len(req.interacted.articles) < 250
        # make sure data is consistent
        assert len(req.interest_profile.click_history) == len(req.interacted.articles)
