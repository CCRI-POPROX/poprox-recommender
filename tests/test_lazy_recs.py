import pytest

from poprox_recommender.lazy_recs import LazyRecs


def test_fill_exact_position():
    recs = LazyRecs(num_slots=3)
    recs.multifill(["b", "a"])
    recs.fill_position(2, "c")
    assert recs.items == ["b", "c", "a"]


def test_fill_nonexistent_slot():
    recs = LazyRecs(num_slots=1)
    with pytest.raises(ValueError):
        recs.fill_position(2, "a")


def test_fill_slot_twice():
    recs = LazyRecs(num_slots=1)
    with pytest.raises(ValueError):
        recs.fill_position(1, "a")
        recs.fill_position(1, "b")


def test_fill_slots_in_order():
    recs = LazyRecs(num_slots=3)
    recs.fill_next("a")
    recs.fill_next("b")
    recs.fill_next("c")
    assert recs.items == ["a", "b", "c"]


def test_filling_slots_incompletely():
    recs = LazyRecs(num_slots=3)
    recs.fill_next("a")
    recs.fill_next("b")
    with pytest.raises(RuntimeError):
        assert recs.items == ["a", "b"]


def test_filling_too_many_slots():
    recs = LazyRecs(num_slots=1)
    recs.fill_next("a")
    with pytest.raises(ValueError):
        recs.fill_next("b")


def test_multifill():
    recs = LazyRecs(num_slots=3)
    recs.multifill(["a", "b", "c"])
    assert recs.items == ["a", "b", "c"]


def test_multifill_by_score():
    items = ["a", "b", "c", "d", "e"]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    recs = LazyRecs(num_slots=3)
    recs.multifill_by_score(items, scores)
    assert recs.items == ["e", "d", "c"]


def test_multifill_by_score_already_partially_filled():
    items = ["a", "b", "c", "d", "e"]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    recs = LazyRecs(num_slots=3)
    recs.fill_position(2, "a")
    recs.multifill_by_score(items, scores)
    assert recs.items == ["e", "a", "d"]
