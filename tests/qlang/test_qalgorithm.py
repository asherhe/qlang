"""Specification cases for the Grover search entry point."""

import pytest

from qlang.evaluate import evaluate

from qlang.qalgorithm import search

missing_phase_output = pytest.mark.xfail(
    strict=True, raises=ValueError, reason="a bare Bit oracle has no output entry for Grover"
)
undecomposed_equality = pytest.mark.xfail(
    strict=True, raises=ValueError, reason="search passes EQ to compile_oracle before decomposition"
)


@missing_phase_output
def test_search_returns_satisfying_assignments(p):
    assert search({"p": [0, 1]}, p) == {"p": 1}


@missing_phase_output
def test_search_returns_none_when_unsatisfied(p):
    assert search({"p": [0]}, p) is None

@undecomposed_equality
def test_grover_algorithm_addition_condition(x, y):
    pred = 3 * x + y == 5

    result = search({"x": range(8), "y": range(5)}, pred)

    assert result is not None
    assert evaluate(pred, result)

@undecomposed_equality
def test_grover_algorithm_subtraction_condition(x, y):
    pred = 3 * x - y == 5

    result = search({"x": range(8), "y": range(5)}, pred)

    assert result is not None
    assert evaluate(pred, result)

@undecomposed_equality
def test_grover_algorithm_multiplication_condition(x, y): 
    pred = x * y == 6

    result = search({"x": range(2), "y": range(3)}, pred)

    assert result is not None
    assert evaluate(pred, result)

@undecomposed_equality
def test_grover_algorithm_no_integer_solution(x, y): 
    pred = 3 * x * y == 5

    result = search({"x": range(8), "y": range(5)}, pred)

    assert result is None
