"""Specification cases for the Grover search entry point."""

import pytest

from qlang.evaluate import evaluate

from qlang.qalgorithm import search

needs_mul = pytest.mark.xfail(
    strict=True, raises=TypeError, reason="compile_oracle has no MUL gadget yet"
)


def test_search_returns_satisfying_assignments(p):
    assert search({"p": [0, 1]}, p) == {"p": 1}


def test_search_returns_none_when_unsatisfied(p):
    assert search({"p": [0]}, p) is None

def test_grover_algorithm_addition_condition(x, y):
    pred = x + y == 5

    result = search({"x": range(8), "y": range(4)}, pred)

    assert result is not None
    assert evaluate(pred, result)

def test_grover_algorithm_subtraction_condition(x, y):
    pred = x - y == 5

    result = search({"x": range(8), "y": range(4)}, pred)

    assert result is not None
    assert evaluate(pred, result)

@needs_mul
def test_grover_algorithm_multiplication_condition(x, y): 
    pred = x * y == 6

    result = search({"x": range(8), "y": range(4)}, pred)

    assert result is not None
    assert evaluate(pred, result)

def test_grover_algorithm_no_integer_solution(x, y): 
    # largest possible value is 3 * 7 + 3 = 24
    pred = 3 * x + y == 25

    result = search({"x": range(8), "y": range(4)}, pred)

    assert result is None