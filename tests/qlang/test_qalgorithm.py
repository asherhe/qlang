"""Small specification cases for the currently unfinished search entry point."""

import pytest

from qlang.evaluate import evaluate

from qlang.qalgorithm import search

def test_search_returns_satisfying_assignments(p):
    assert search({"p": [0, 1]}, p) == {"p": 1}


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="search does not yet implement empty solution results")
def test_search_returns_empty_list_when_unsatisfied(p):
    assert search({"p": [0]}, p) == None

def test_grover_algorithm_addition_condition(x, y):
    pred = 3 * x + y == 5

    result = search({"x": range(8), "y": range(5)}, pred)

    assert result is not None
    assert evaluate(pred, result)

def test_grover_algorithm_subtraction_condition(x, y):
    pred = 3 * x - y == 5

    result = search({"x": range(8), "y": range(5)}, pred)

    assert result is not None
    assert evaluate(pred, result)

def test_grover_algorithm_multiplication_condition(x, y): 
    pred = x * y == 6

    result = search({"x": range(2), "y": range(3)}, pred)

    assert result is not None
    assert evaluate(pred, result)

def test_grover_algorithm_no_integer_solution(x, y):
    pred = 3 * x * y == 5

    result = search({"x": range(8), "y": range(5)}, pred)

    assert result is None
