import torch as th
from pytest import raises

from poprox_recommender.torch.datachecks import assert_tensor_size


def test_tensor_size_ok():
    assert_tensor_size(th.randn(10, 50), 10, 50)


def test_tensor_size_fail():
    with raises(AssertionError, match=r"^unexpected tensor size"):
        assert_tensor_size(th.randn(10, 40), 10, 50)


def test_tensor_size_fail_label():
    with raises(AssertionError, match=r"^test: unexpected tensor size"):
        assert_tensor_size(th.randn(10, 40), 10, 50, label="test")


def test_tensor_size_prefix():
    assert_tensor_size(th.randn(10, 50, 25), 10, 50, label="test")


def test_tensor_size_no_prefix():
    with raises(AssertionError, match=r"^test: unexpected tensor size"):
        assert_tensor_size(th.randn(10, 50, 25), 10, 50, label="test", prefix=False)
