"""Small specification cases for the currently unfinished search entry point."""

import pytest

from qlang.qalgorithm import search


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="search is an unimplemented stub returning None")
def test_search_returns_satisfying_assignments(p):
    assert search({"p": [0, 1]}, p) == [{"p": 1}]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="search does not yet implement empty solution results")
def test_search_returns_empty_list_when_unsatisfied(p):
    assert search({"p": [0]}, p) == []
