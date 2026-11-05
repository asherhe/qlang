"""Rewrite behavior and semantic preservation on small complete input domains."""

import itertools
import operator

import pytest

from qlang.evaluate import evaluate
from qlang.expression import Bit, BitType, UIntType, bits_uint, const_bit, const_uint, var_bit, var_uint
from qlang.optimize import RulePass, decompose, optimize_expr, run_rule_pass, traverse_transform


@pytest.mark.parametrize("value", [False, True])
def test_constant_not_folds(value):
    result = optimize_expr(~const_bit(value))
    assert result.bit_type == BitType.CONST
    assert result.value is (not value)


@pytest.mark.parametrize("op", [operator.and_, operator.or_, operator.xor])
@pytest.mark.parametrize("a,b", list(itertools.product([False, True], repeat=2)))
def test_constant_boolean_folding(op, a, b):
    result = optimize_expr(op(const_bit(a), const_bit(b)))
    assert result.bit_type == BitType.CONST
    assert result.value is op(a, b)


@pytest.mark.parametrize("build", [
    lambda p, q: ~~p,
    lambda p, q: p & True,
    lambda p, q: False & p,
    lambda p, q: p ^ False,
    lambda p, q: True ^ p,
    lambda p, q: p & p,
    lambda p, q: p ^ p,
    lambda p, q: p & ~p,
    lambda p, q: ~p ^ p,
    lambda p, q: p | ~p,
    lambda p, q: (p | q) & ~(p & q),
], ids=["double-not", "and-one", "and-zero", "xor-zero", "xor-one", "and-self", "xor-self", "and-complement", "xor-complement", "or-complement", "nested-logic"])
def test_boolean_rewrites_preserve_truth_table(p, q, build):
    expr = build(p, q)
    result = optimize_expr(expr)
    for a, b in itertools.product([False, True], repeat=2):
        assignment = {"p": a, "q": b}
        assert evaluate(result, assignment) is evaluate(expr, assignment)


@pytest.mark.parametrize("op", [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge])
def test_uint_comparison_rewrites_preserve_values(x, y, op):
    expr = op(x, y)
    result = optimize_expr(expr)
    for a, b in itertools.product(range(8), range(4)):
        assert evaluate(result, {"x": a, "y": b}) is op(a, b)


@pytest.mark.parametrize("constant", [0, 1, 6, 7, 8])
@pytest.mark.parametrize("constant_first", [False, True])
def test_comparison_constant_bounds(x, constant, constant_first):
    expr = (constant < x) if constant_first else (x < constant)
    result = optimize_expr(expr)
    for value in range(8):
        assert evaluate(result, {"x": value}) is evaluate(expr, {"x": value})


@pytest.mark.parametrize("op,expected", [(operator.add, 5), (operator.mul, 6), (operator.rshift, 0)])
def test_constant_arithmetic_fold(op, expected):
    result = optimize_expr(op(const_uint(2, 4), const_uint(3, 4)))
    assert result.uint_type == UIntType.CONST
    assert result.const_value == expected


@pytest.mark.parametrize("a,b,expected", [
    (3, 1, 2),
    (1, 3, 14),
    (0, 0, 0),
])
def test_subtraction_constant_fold(a, b, expected):
    result = optimize_expr(const_uint(a, 4) - const_uint(b, 4))
    assert result.const_value == expected
    assert result.width == 4


@pytest.mark.parametrize("value,amount", [
    (1, 1), (3, 1),
    (7, 1), (1, 3),
])
def test_constant_left_shift_wraps(value, amount):
    result = optimize_expr(const_uint(value, 3) << amount)
    assert result.const_value == (value << amount) & 0b111


@pytest.mark.parametrize("build", [
    lambda x, y: x + 0,
    lambda x, y: x - 0,
    lambda x, y: x - x,
    lambda x, y: (x + y) - y,
    lambda x, y: x * 0,
    lambda x, y: x * 1,
    lambda x, y: x * 4,
    lambda x, y: (x + 2) + 3,
    lambda x, y: (x * 3) * 4,
    lambda x, y: x << 0,
    lambda x, y: x >> 0,
    lambda x, y: x << 3,
    lambda x, y: x >> 3,
    lambda x, y: (x << 1) << 1,
    lambda x, y: (x >> 1) >> 1,
], ids=["add-zero", "sub-zero", "sub-self", "cancel-addend", "mul-zero", "mul-one", "mul-power-two", "add-reassociate", "mul-reassociate", "lshift-zero", "rshift-zero", "lshift-overshift", "rshift-overshift", "merge-left-shifts", "merge-right-shifts"])
def test_arithmetic_rewrites_preserve_values(x, y, build):
    expr = build(x, y)
    result = optimize_expr(expr)
    for a, b in itertools.product(range(8), range(4)):
        assert evaluate(result, {"x": a, "y": b}) == evaluate(expr, {"x": a, "y": b})


def test_add_self_preserves_carry(x):
    result = optimize_expr(x + x)
    assert evaluate(result, {"x": 7}) == 14


@pytest.mark.parametrize("build,assignment,expected", [
    (lambda x, y: (0 - (x + 0)), {"x": 1, "y": 0}, 15),
    (lambda x, y: (0 - (x * 1)), {"x": 1, "y": 0}, 15),
    (lambda x, y: ((x + y) - y) << 3, {"x": 1, "y": 0}, 8),
    (lambda x, y: (const_uint(1, 4) + const_uint(0, 4)) << 3, {}, 8),
    (lambda x, y: (const_uint(1, 4) >> 1) - const_uint(1), {}, 15),
], ids=["add-identity-before-underflow", "mul-identity-before-underflow", "cancel-before-shift", "constant-add-before-shift", "constant-rshift-before-underflow"])
def test_rewrites_preserve_width_sensitive_parent_semantics(x, y, build, assignment, expected):
    expr = build(x, y)
    assert evaluate(expr, assignment) == expected
    assert evaluate(optimize_expr(expr), assignment) == expected


def test_add_self_becomes_shifted_bits():
    node = var_uint("x", 3)
    result = optimize_expr(node + node)
    assert result.uint_type == UIntType.BITS
    assert result.width == 4
    assert result.bits == (const_bit(0),) + node.bits


def test_mul_by_power_of_two_becomes_shifted_bits_padded_to_product_width():
    node = var_uint("x", 3)
    expr = node * 4  # constant 4 is 3 bits wide, so the product is 6 bits wide
    result = optimize_expr(expr)
    assert result.uint_type == UIntType.BITS
    assert result.width == expr.width == 6
    zero = const_bit(0)
    assert result.bits == (zero, zero) + node.bits + (zero,)


def test_constant_shift_truncates_to_shift_node_width_not_operand_width(x, y):
    # (x + y) - y is 4 bits wide; shifting it left by 3 must keep the carry bit, not drop to 3
    expr = ((x + y) - y) << 3
    assert evaluate(expr, {"x": 1, "y": 0}) == 8
    assert evaluate(optimize_expr(expr), {"x": 1, "y": 0}) == 8


def test_optimizer_does_not_mutate_original_expression(x):
    expr = x + 0
    operands = expr.operands
    width = expr.width
    optimize_expr(expr)
    assert expr.operands is operands
    assert expr.width == width
    assert expr.uint_type == UIntType.ADD


def test_leaf_identity(x, p):
    assert optimize_expr(x) is x
    assert optimize_expr(p) is p


def test_or_decomposition_removes_or(p, q):
    result = run_rule_pass(p | q, decompose)
    assert result.bit_type == BitType.NOT
    assert result.operands[0].bit_type == BitType.AND


def test_rule_registry_distinguishes_intenum_classes():
    rules = RulePass("test")

    @rules.rule(BitType.NOT)
    def bit_rule(*args):
        return None

    @rules.rule(UIntType.ADD)
    def uint_rule(*args):
        return None

    assert int(BitType.NOT) == int(UIntType.ADD)
    assert rules.rules_for(BitType.NOT) == [bit_rule]
    assert rules.rules_for(UIntType.ADD) == [uint_rule]


def test_rule_registration_order_first_match_wins(p, q):
    rules = RulePass("ordered")
    visited = []

    @rules.rule(BitType.AND)
    def first(node, a, b):
        visited.append("first")
        return const_bit(False)

    @rules.rule(BitType.AND)
    def second(node, a, b):
        visited.append("second")
        return const_bit(True)

    assert run_rule_pass(p & q, rules).value is False
    assert visited == ["first"]


def test_traversal_visits_shared_nodes_once_bottom_up(p, q):
    shared = p & q
    root = shared ^ shared
    visited = []

    def record(node):
        visited.append(node)
        return node

    assert traverse_transform(root, record) is root
    assert len(visited) == 2
    assert visited[0] is shared and visited[1] is root


def _bit_of(uint, index):
    """the Bit that selects position index of a UInt, built the way a UInt owns its bits"""
    return Bit._create(bit_type=BitType.UINT, index=index, parent_uint=uint)


def test_bits_of_bits_route_to_original_sources():
    p, q, r = var_bit("p"), var_bit("q"), var_bit("r")
    inner = bits_uint((p, q, r))
    outer = bits_uint((_bit_of(inner, 2), _bit_of(inner, 0), r))
    result = optimize_expr(outer)
    assert result.uint_type == UIntType.BITS
    assert result.bits == (r, p, r)


def test_bits_of_bits_route_through_several_levels():
    p, q = var_bit("p"), var_bit("q")
    first = bits_uint((p, q))
    second = bits_uint((_bit_of(first, 1), _bit_of(first, 0)))
    third = bits_uint((_bit_of(second, 1), _bit_of(second, 0)))
    assert optimize_expr(third).bits == (p, q)


def test_bits_of_non_bits_uint_are_not_rerouted():
    x = var_uint("x", 2)
    node = bits_uint((x.bits[1], x.bits[0]))
    assert optimize_expr(node) is node
