"""Classical semantics, validation, width wrapping, and evaluation memoization."""

import importlib
import itertools
import operator

import pytest

from qlang.evaluate import evaluate
from qlang.expression import Bit, BitType, UInt, UIntType, const_bit, const_uint


@pytest.mark.parametrize("value", [False, True])
def test_bit_constants(value):
    assert evaluate(const_bit(value), {}) is value


@pytest.mark.parametrize("value,width", [(0, 1), (1, 1), (6, 3), (15, 4)])
def test_uint_constants(value, width):
    assert evaluate(const_uint(value, width), {}) == value


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize("a,b", list(itertools.product([False, True], repeat=2)))
def test_boolean_truth_tables(p, q, op, a, b):
    assert evaluate(op(p, q), {"p": a, "q": b}) is op(a, b)


@pytest.mark.parametrize("a,b", list(itertools.product([False, True], repeat=2)))
def test_nested_boolean_and_not(p, q, a, b):
    assert evaluate(~(p & q) | (p ^ q), {"p": a, "q": b}) is (
        not (a and b) or (a ^ b)
    )


@pytest.mark.parametrize("op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge])
@pytest.mark.parametrize("a,b", [(0, 0), (1, 2), (3, 3), (7, 1)])
def test_uint_comparisons(x, y, op, a, b):
    assert evaluate(op(x, y), {"x": a, "y": b}) is op(a, b)


@pytest.mark.parametrize("op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge])
def test_bit_comparisons(p, q, op):
    for a, b in itertools.product([False, True], repeat=2):
        assert evaluate(op(p, q), {"p": a, "q": b}) is op(a, b)


@pytest.mark.parametrize("a,b", [(0, 0), (1, 3), (7, 3), (3, 1)])
def test_arithmetic_and_fixed_width_subtraction(x, y, a, b):
    assignments = {"x": a, "y": b}
    assert evaluate(x + y, assignments) == a + b
    assert evaluate(x * y, assignments) == a * b
    assert evaluate(x - y, assignments) == (a - b) % 8


def test_four_bit_twos_complement_underflow(make_uint):
    a, b = make_uint("a", 4), make_uint("b", 4)
    assert evaluate(a - b, {"a": 3, "b": 5}) == 14
    assert evaluate((a - b) < 3, {"a": 3, "b": 5}) is False


@pytest.mark.parametrize("value,amount", [(0, 0), (7, 0), (3, 1), (7, 2), (7, 3), (7, 8)])
def test_fixed_width_shifts(x, value, amount):
    assert evaluate(x << amount, {"x": value}) == (value << amount) % 8
    assert evaluate(x >> amount, {"x": value}) == value >> amount


def test_large_shift_short_circuits(x, make_uint):
    amount = make_uint("amount", 64)
    assert evaluate(x << amount, {"x": 1, "amount": 2**63}) == 0


def test_nested_arithmetic_and_reflected_operations(x, y):
    assignments = {"x": 6, "y": 2}
    assert evaluate((x + y) * 3 == 24, assignments) is True
    assert evaluate(2 + x, assignments) == 8
    assert evaluate(2 * x, assignments) == 12
    assert evaluate(1 - x, assignments) == 3
    assert evaluate(1 << y, assignments) == 0  # left operand has width one
    assert evaluate(8 >> y, assignments) == 2


@pytest.mark.parametrize("value", [0, 1, 2, 5, 7])
def test_indexed_bits_and_input_immutability(x, value):
    assignments = {"x": value}
    for i, bit in enumerate(x.bits):
        assert evaluate(bit, assignments) is bool((value >> i) & 1)
    assert evaluate(x, assignments) == value
    assert assignments == {"x": value}


def test_indexed_bit_and_parent_in_same_predicate(x):
    assert evaluate(x.bits[1] & (x == 2), {"x": 2}) is True


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("value", [-1, 8, 1.5])
def test_reject_invalid_integer_assignments(x, indexed, value):
    with pytest.raises(ValueError):
        evaluate(x.bits[0] if indexed else x, {"x": value})


def test_missing_assignment_in_nested_expression(x):
    with pytest.raises(ValueError, match="not assigned"):
        evaluate((x + 1) == 2, {})


@pytest.mark.parametrize("value,expected", [(0, False), (1, True), (2, True)])
def test_standalone_bits_normalize_without_mutating(p, value, expected):
    assignments = {"p": value}
    assert evaluate(p, assignments) is expected
    assert assignments == {"p": value}


@pytest.mark.parametrize("kind,operands", [(BitType.NOT, ()), (BitType.NOT, (True, True)), (BitType.AND, (True,)), (BitType.AND, (True, True, True))])
def test_invalid_bit_arity(kind, operands):
    node = Bit(bit_type=kind, operands=tuple(const_bit(v) for v in operands))
    with pytest.raises(ValueError):
        evaluate(node, {})


def test_invalid_uint_arity(x):
    node = UInt(uint_type=UIntType.ADD, width=4, operands=(x,))
    with pytest.raises(ValueError):
        evaluate(node, {"x": 1})


def test_wrong_operand_type_is_rejected(x, p):
    node = Bit(bit_type=BitType.AND, operands=(x, p))
    with pytest.raises(ValueError):
        evaluate(node, {"x": 1, "p": True})


def test_missing_constant_value():
    with pytest.raises(ValueError, match="missing value"):
        evaluate(Bit(bit_type=BitType.CONST), {})


def test_memo_is_fresh_for_each_assignment(x):
    expr = (x + 1) * (x + 1)
    assert evaluate(expr, {"x": 1}) == 4
    assert evaluate(expr, {"x": 2}) == 9


def test_shared_nodes_are_computed_once_including_not(p, monkeypatch):
    module = importlib.import_module("qlang.evaluate")
    original = module._evaluate_uncached
    counts = {}

    def counted(node, assignments, memo):
        counts[id(node)] = counts.get(id(node), 0) + 1
        return original(node, assignments, memo)

    shared = p & p
    expr = (~shared) | shared
    monkeypatch.setattr(module, "_evaluate_uncached", counted)
    assert evaluate(expr, {"p": True}) is True
    assert counts[id(shared)] == 1
