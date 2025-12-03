import logging
from datetime import timedelta
from itertools import islice

from pytest import fixture, skip

from poprox_recommender.data.poprox import PoproxData, SlateSet

logger = logging.getLogger(__name__)


@fixture(scope="module")
def poprox_data():
    yield _load_or_skip()


def _load_or_skip(slates: SlateSet = "all"):
    try:
        return PoproxData(slates=slates)
    except FileNotFoundError as e:
        logger.error("POPROX not found: %r", e)
        skip("POPROX data not available")


def test_load(poprox_data: PoproxData):
    "Test the data loads & basic properties work."
    assert poprox_data.n_articles > 100
    assert poprox_data.n_requests > 500


def test_scan_slate_ids(poprox_data: PoproxData):
    "Test that the slate ID iterator is consistent with request count."
    # count the number of items in the iterator
    count = sum(1 for _ in poprox_data.iter_slate_ids())
    assert count == poprox_data.n_requests


def test_lookup_requests(poprox_data: PoproxData):
    "Test that we can look up some requests."

    for i, slate_id in enumerate(islice(poprox_data.iter_slate_ids(), 0, 5000, 100)):
        try:
            req = poprox_data.lookup_request(slate_id)
            assert req is not None
            # some basic sanity checks
            assert 10 < len(req.candidates.articles) < 150
            assert 0 <= len(req.interacted.articles) < 100

            slate_ts = req.interest_profile.model_extra["slate_created_at"]

            # make sure data is consistent
            click_aids = set(c.article_id for c in req.interest_profile.click_history)
            assert len(click_aids) == len(req.interacted.articles)

            # check the dates of candidate articles
            times = [a.created_at for a in req.candidates.articles]
            assert not any(t is None for t in times)
            # are all articles ingested on the same day?
            min_ts = min(times)
            max_ts = max(times)
            assert (max_ts - min_ts) <= timedelta(days=1)
            # are all articles ingested after they are published, and before the newsletter?
            for a in req.candidates.articles:
                assert a.published_at <= a.created_at
                assert a.created_at <= slate_ts
        except Exception as e:
            e.add_note(f"Error occurred in test slate {i} ({slate_id})")
            raise e


def test_lookup_truth(poprox_data: PoproxData):
    "Test that we can look up some truth."

    n = 0
    for i, slate_id in enumerate(islice(poprox_data.iter_slate_ids(), 0, 5000, 15)):
        try:
            truth = poprox_data.slate_truth(slate_id)
            assert truth is not None
            if len(truth) == 0:
                continue

            n += 1
            req = poprox_data.lookup_request(slate_id)
            slate_ts = req.interest_profile.slate_created_at
            for aid in truth.index:
                art = poprox_data.lookup_article(aid, source="clicked")
                assert art.article_id == aid

                # was the article published before the newsletter?
                assert art.published_at <= slate_ts
                # was the article ingested no more than 1 day before the
                # newsletter? it may be ingested after the newsletter, if we
                # re-ingested it on a later day, so we don't test for that.
                assert slate_ts - art.created_at <= timedelta(days=1)

                # TODO: when we have explicit candidate sets, test that clicked articles
                # are in the candidate set.
        except Exception as e:
            e.add_note(f"Error occurred in test slate {i} ({slate_id})")
            raise e

    # make sure we got some articles with nonzero truth so the test isn't void
    assert n > 10, "no test requests had clicks"


def test_request_articles(poprox_data: PoproxData):
    "Look for articles mentioned by requests."
    if not hasattr(poprox_data, "lookup_article"):
        skip("testing against old PoproxData")

    for slate_id in islice(poprox_data.iter_slate_ids(), 15, 5000, 150):
        req = poprox_data.lookup_request(slate_id)
        assert req is not None

        click_aids = set(c.article_id for c in req.interest_profile.click_history)
        assert len(click_aids) == len(req.interacted.articles)

        # check that we can look up all the specified articles
        for cand in req.candidates.articles:
            art = poprox_data.lookup_article(cand.article_id)
            assert art.article_id == cand.article_id
            assert art.headline == cand.headline
            # are all articles ingested after they are published?
            for art in req.candidates.articles:
                assert art.published_at <= art.created_at

        for click in req.interest_profile.click_history:
            art = poprox_data.lookup_article(click.article_id, source="clicked")
            assert art.article_id == click.article_id
            # are all articles ingested after they are published?
            for art in req.candidates.articles:
                assert art.published_at <= art.created_at


def test_latest_slates():
    poprox_data = _load_or_skip("latest")

    slates = list(poprox_data.iter_slate_ids())
    db = poprox_data.duck.cursor()
    db.execute("SELECT COUNT(DISTINCT account_id) FROM newsletters")
    (n,) = db.fetchone() or [0]
    # did we get one slate per account?
    assert len(slates) == n


def test_recent_slates():
    poprox_data = _load_or_skip("recent")

    slates = list(poprox_data.iter_slate_ids())
    db = poprox_data.duck.cursor()
    db.execute("SELECT COUNT(DISTINCT account_id) FROM newsletters")
    (n,) = db.fetchone() or [0]
    # did we get one slate per account?
    assert len(slates) <= n
