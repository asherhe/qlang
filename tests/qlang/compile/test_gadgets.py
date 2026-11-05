"""Gadget execution tests: each @gadget-dispatched operator compiles to a valid phase oracle."""

import itertools
from collections import defaultdict

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlang.compile import Oracle
from qlang.compile.compile import compile_oracle, parse_input
from qlang.compile.gadgets import _GADGETS
from qlang.evaluate import evaluate
from qlang.expression import (
    ASTNode,
    Bit,
    BitType,
    UInt,
    UIntType,
    bits_uint,
    const_bit,
    const_uint,
    var_bit,
    var_uint,
)


@pytest.fixture
def isolated_registry():
    """save and restore the global gadget registry around each test"""
    saved = dict(_GADGETS)
    try:
        yield
    finally:
        _GADGETS.clear()
        _GADGETS.update(saved)


def assert_phase_oracle(expr, expected_fn=None, cases=None):
    """Compare the compiled circuit phases with evaluation of the original expression.

    expected_fn, if given, is a plain-python function of the input assignments returning the
    predicate's truth value. Otherwise the expression is evaluated with qlang.evaluate.
    cases, if given, is a list of input assignments to check. Sweeping all 2^n inputs is slow,
    so callers pass hard-coded cases for anything beyond a few bits. The uniform-superposition
    check only runs when every input is swept (cases is None).
    Variables the optimizer removes (e.g. a ^ a) are read as 0, since they cannot affect the result.
    """
    # Gather the original variables before optimization can remove them.
    reference_widths = {}
    for node in parse_input(expr):
        if isinstance(node, UInt) and node.uint_type == UIntType.VAR:
            reference_widths[node.name] = node.width
        elif isinstance(node, Bit) and node.bit_type == BitType.VAR:
            reference_widths[node.name] = 1

    oracle = compile_oracle(expr)

    # Contract assertions: compile_oracle's phase-only interface.
    assert isinstance(oracle, Oracle)
    assert oracle.output == []
    assert oracle.dirty_ancillas == []
    input_qubits = [qubit for register in oracle.inputs.values() for qubit in register]
    assert len(input_qubits) == len(set(input_qubits))
    assert len(oracle.clean_ancillas) == len(set(oracle.clean_ancillas))
    assert not (set(input_qubits) & set(oracle.clean_ancillas))
    assert (set(input_qubits) | set(oracle.clean_ancillas)) == set(range(oracle.circuit.num_qubits))

    for name, register in oracle.inputs.items():
        assert len(register) == reference_widths[name]

    names = list(reference_widths)
    sweep_all = cases is None
    if sweep_all:
        domains = [range(1 << reference_widths[name]) for name in names]
        cases = [dict(zip(names, values)) for values in itertools.product(*domains)]
    expected_superposition = np.zeros(2 ** oracle.circuit.num_qubits, dtype=complex)

    for case in cases:
        assignments = {name: case[name] for name in names}
        basis = sum(
            ((assignments[name] >> bit) & 1) << qubit
            for name, register in oracle.inputs.items()
            for bit, qubit in enumerate(register)
        )
        if expected_fn is not None:
            truth = expected_fn(defaultdict(int, assignments))
        else:
            truth = evaluate(expr, assignments)
        sign = -1 if truth else 1

        expected = np.zeros_like(expected_superposition)
        expected[basis] = sign
        # Exact positions and signs detect input changes, leftover scratch,
        # and wrong global signs on constant predicates as well.
        actual = Statevector.from_int(basis, len(expected)).evolve(oracle.circuit)
        np.testing.assert_allclose(actual.data, expected, atol=1e-12)
        expected_superposition[basis] = sign / np.sqrt(2 ** len(input_qubits))

    if sweep_all:
        # Uniform superposition over all inputs must evolve to the signed superposition.
        prepared = QuantumCircuit(oracle.circuit.num_qubits)
        for qubit in input_qubits:
            prepared.h(qubit)
        prepared.compose(oracle.circuit, inplace=True)
        np.testing.assert_allclose(Statevector.from_instruction(prepared).data, expected_superposition, atol=1e-12)
    return oracle


def test_and(isolated_registry):
    ASTNode.reset()
    a, b = var_bit("a"), var_bit("b")
    assert_phase_oracle(a & b)


def test_or(isolated_registry):
    ASTNode.reset()
    a, b = var_bit("a"), var_bit("b")
    assert_phase_oracle(a | b)


def test_xor(isolated_registry):
    ASTNode.reset()
    a, b = var_bit("a"), var_bit("b")
    assert_phase_oracle(a ^ b)


def test_not(isolated_registry):
    ASTNode.reset()
    a = var_bit("a")
    assert_phase_oracle(~a)


def test_compound_boolean(isolated_registry):
    ASTNode.reset()
    a, b, c = var_bit("a"), var_bit("b"), var_bit("c")
    assert_phase_oracle((a | b) & ~c)


def test_shared_subexpression(isolated_registry):
    ASTNode.reset()
    a, b, c = var_bit("a"), var_bit("b"), var_bit("c")
    shared = a | b
    assert_phase_oracle(shared ^ (shared & c))


def test_repeated_operands(isolated_registry):
    ASTNode.reset()
    a = var_bit("a")
    for expr in (a & a, a | a, a ^ a):
        assert_phase_oracle(expr)


def test_const_leaves(isolated_registry):
    ASTNode.reset()
    assert_phase_oracle(const_bit(True))
    assert_phase_oracle(const_bit(False))
    assert_phase_oracle(var_bit("a") & const_bit(True))


def test_leaf_roots_and_constant_folding(isolated_registry):
    ASTNode.reset()
    a, b = var_bit("a"), var_bit("b")
    assert_phase_oracle(a)
    assert_phase_oracle(a & const_bit(False))
    assert_phase_oracle((a & const_bit(True)) | (b & const_bit(True)))


def test_bit_comparisons(isolated_registry):
    ASTNode.reset()
    a, b = var_bit("a"), var_bit("b")
    assert_phase_oracle(a != b)
    assert_phase_oracle(a < b)


def test_indexed_uint_bit(isolated_registry):
    ASTNode.reset()
    x = var_uint("x", 3)
    oracle = assert_phase_oracle(x.bits[2] & x.bits[0])
    assert len(oracle.inputs["x"]) == 3
    assert len(oracle.clean_ancillas) == 1


CASES_TEST_SINGLE_INDEXED_BIT = [{"x": 0}, {"x": 2}, {"x": 5}, {"x": 7}]


def test_single_indexed_bit_preserves_full_parent_width(isolated_registry):
    ASTNode.reset()
    x = var_uint("x", 3)
    oracle = assert_phase_oracle(x.bits[1], cases=CASES_TEST_SINGLE_INDEXED_BIT)
    assert len(oracle.inputs["x"]) == 3
    assert oracle.clean_ancillas == []


# ---- UInt gadgets: each is checked through a != comparison so the predicate is a single Bit ----


CASES_TEST_ADD = [{"a": 0, "b": 0, "c": 0}, {"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 3, "c": 6}, {"a": 3, "b": 1, "c": 4}, {"a": 2, "b": 1, "c": 5}, {"a": 3, "b": 3, "c": 5}, {"a": 1, "b": 1, "c": 3}]


def test_add(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 2), var_uint("b", 2), var_uint("c", 3)
    assert_phase_oracle((a + b) == c, lambda v: v["a"] + v["b"] == v["c"], cases=CASES_TEST_ADD)


CASES_TEST_ADD_MISMATCHED_WIDTHS = [{"a": 0, "b": 0, "c": 0}, {"a": 7, "b": 1, "c": 8}, {"a": 7, "b": 0, "c": 7}, {"a": 5, "b": 1, "c": 6}, {"a": 3, "b": 1, "c": 5}]


def test_add_mismatched_widths(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 1), var_uint("c", 4)
    assert_phase_oracle((a + b) == c, lambda v: v["a"] + v["b"] == v["c"], cases=CASES_TEST_ADD_MISMATCHED_WIDTHS)


CASES_TEST_ADD_CONST = [{"a": 0, "c": 3}, {"a": 1, "c": 4}, {"a": 2, "c": 5}, {"a": 3, "c": 6}, {"a": 3, "c": 5}, {"a": 0, "c": 0}, {"a": 1, "c": 3}]


def test_add_const(isolated_registry):
    ASTNode.reset()
    a, c = var_uint("a", 2), var_uint("c", 3)
    assert_phase_oracle((a + 3) == c, lambda v: v["a"] + 3 == v["c"], cases=CASES_TEST_ADD_CONST)


CASES_TEST_ADDITION_SHARED_BITS = [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}]


def test_addition_with_shared_input_bits(isolated_registry):
    ASTNode.reset()
    x = var_uint("x", 2)
    assert_phase_oracle(x + bits_uint(x.bits) == 2, cases=CASES_TEST_ADDITION_SHARED_BITS)


CASES_TEST_SUB = [{"a": 3, "b": 1, "c": 2}, {"a": 0, "b": 1, "c": 3}, {"a": 2, "b": 2, "c": 0}, {"a": 1, "b": 0, "c": 1}, {"a": 0, "b": 0, "c": 1}, {"a": 2, "b": 3, "c": 2}, {"a": 3, "b": 3, "c": 3}]


def test_sub(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 2), var_uint("b", 2), var_uint("c", 2)
    assert_phase_oracle((a - b) == c, lambda v: (v["a"] - v["b"]) % 4 == v["c"], cases=CASES_TEST_SUB)


CASES_TEST_SUB_MISMATCHED_WIDTHS = [{"a": 0, "b": 1, "c": 7}, {"a": 7, "b": 0, "c": 7}, {"a": 4, "b": 1, "c": 3}, {"a": 0, "b": 1, "c": 0}, {"a": 7, "b": 1, "c": 6}]


def test_sub_mismatched_widths(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 1), var_uint("c", 3)
    assert_phase_oracle((a - b) == c, lambda v: (v["a"] - v["b"]) % 8 == v["c"], cases=CASES_TEST_SUB_MISMATCHED_WIDTHS)


CASES_TEST_LSHIFT_CONST = [{"a": 0, "c": 0}, {"a": 3, "c": 6}, {"a": 5, "c": 2}, {"a": 7, "c": 6}, {"a": 4, "c": 0}, {"a": 7, "c": 7}, {"a": 1, "c": 1}]


def test_lshift_const(isolated_registry):
    ASTNode.reset()
    a, c = var_uint("a", 3), var_uint("c", 3)
    assert_phase_oracle((a << 1) == c, lambda v: (v["a"] << 1) % 8 == v["c"], cases=CASES_TEST_LSHIFT_CONST)


CASES_TEST_LSHIFT_VARIABLE = [{"a": 1, "b": 2, "c": 4}, {"a": 5, "b": 1, "c": 2}, {"a": 7, "b": 3, "c": 0}, {"a": 3, "b": 0, "c": 3}, {"a": 3, "b": 0, "c": 0}, {"a": 6, "b": 1, "c": 6}]


@pytest.mark.xfail(reason="shifting by a variable amount is not supported yet", raises=TypeError)
def test_lshift_variable(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 2), var_uint("c", 3)
    assert_phase_oracle((a << b) == c, lambda v: (v["a"] << v["b"]) % 8 == v["c"], cases=CASES_TEST_LSHIFT_VARIABLE)


CASES_TEST_RSHIFT_CONST = [{"a": 0, "c": 0}, {"a": 7, "c": 3}, {"a": 5, "c": 2}, {"a": 1, "c": 0}, {"a": 3, "c": 1}, {"a": 6, "c": 2}, {"a": 4, "c": 4}]


def test_rshift_const(isolated_registry):
    ASTNode.reset()
    a, c = var_uint("a", 3), var_uint("c", 3)
    assert_phase_oracle((a >> 1) == c, lambda v: (v["a"] >> 1) == v["c"], cases=CASES_TEST_RSHIFT_CONST)


CASES_TEST_RSHIFT_VARIABLE = [{"a": 6, "b": 1, "c": 3}, {"a": 7, "b": 3, "c": 0}, {"a": 5, "b": 0, "c": 5}, {"a": 4, "b": 2, "c": 1}, {"a": 4, "b": 2, "c": 2}, {"a": 7, "b": 1, "c": 7}]


@pytest.mark.xfail(reason="shifting by a variable amount is not supported yet", raises=TypeError)
def test_rshift_variable(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 2), var_uint("c", 3)
    assert_phase_oracle((a >> b) == c, lambda v: (v["a"] >> v["b"]) == v["c"], cases=CASES_TEST_RSHIFT_VARIABLE)


CASES_TEST_EQ = [{"a": 0, "b": 0}, {"a": 5, "b": 5}, {"a": 7, "b": 7}, {"a": 0, "b": 1}, {"a": 5, "b": 4}, {"a": 7, "b": 0}, {"a": 3, "b": 4}, {"a": 2, "b": 6}]


def test_eq(isolated_registry):
    ASTNode.reset()
    a, b = var_uint("a", 3), var_uint("b", 3)
    assert_phase_oracle(a == b, lambda v: v["a"] == v["b"], cases=CASES_TEST_EQ)


CASES_TEST_EQ_MISMATCHED_WIDTHS = [{"a": 0, "b": 0}, {"a": 3, "b": 3}, {"a": 2, "b": 2}, {"a": 4, "b": 0}, {"a": 5, "b": 1}, {"a": 7, "b": 3}, {"a": 1, "b": 2}, {"a": 0, "b": 3}]


def test_eq_mismatched_widths(isolated_registry):
    ASTNode.reset()
    a, b = var_uint("a", 3), var_uint("b", 2)
    assert_phase_oracle(a == b, lambda v: v["a"] == v["b"], cases=CASES_TEST_EQ_MISMATCHED_WIDTHS)


CASES_TEST_EQ_CONST = [{"a": 5}, {"a": 0}, {"a": 1}, {"a": 4}, {"a": 6}, {"a": 7}]


def test_eq_const(isolated_registry):
    ASTNode.reset()
    a = var_uint("a", 3)
    assert_phase_oracle(a == const_uint(5, 3), lambda v: v["a"] == 5, cases=CASES_TEST_EQ_CONST)


CASES_TEST_UINT_EQ_OPTIMIZED = [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}]


def test_uint_equality_is_compiled_through_optimization(isolated_registry):
    ASTNode.reset()
    assert_phase_oracle(var_uint("x", 2) == 2, cases=CASES_TEST_UINT_EQ_OPTIMIZED)


CASES_TEST_UNEQUAL_WIDTH_COMPARISON = [{"x": 0, "y": 0}, {"x": 0, "y": 1}, {"x": 0, "y": 2}, {"x": 0, "y": 3}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 1, "y": 2}, {"x": 1, "y": 3}]


def test_uint_comparisons_with_unequal_widths(isolated_registry):
    ASTNode.reset()
    x, y = var_uint("x", 1), var_uint("y", 2)
    # Cover padding on both sides of < and reuse of the same input.
    assert_phase_oracle(x != y, cases=CASES_TEST_UNEQUAL_WIDTH_COMPARISON)
    assert_phase_oracle(x < y, cases=CASES_TEST_UNEQUAL_WIDTH_COMPARISON)
    assert_phase_oracle(y < x, cases=CASES_TEST_UNEQUAL_WIDTH_COMPARISON)
    assert_phase_oracle(y != y, cases=CASES_TEST_UNEQUAL_WIDTH_COMPARISON)
    assert_phase_oracle(y < y, cases=CASES_TEST_UNEQUAL_WIDTH_COMPARISON)


CASES_TEST_COMPARISON_IN_BOOLEAN = [{"x": 0, "enabled": 0}, {"x": 0, "enabled": 1}, {"x": 1, "enabled": 0}, {"x": 1, "enabled": 1}, {"x": 2, "enabled": 0}, {"x": 2, "enabled": 1}, {"x": 3, "enabled": 0}, {"x": 3, "enabled": 1}]


def test_uint_comparison_inside_boolean_expression(isolated_registry):
    ASTNode.reset()
    x, enabled = var_uint("x", 2), var_bit("enabled")
    assert_phase_oracle((x < 2) & enabled, cases=CASES_TEST_COMPARISON_IN_BOOLEAN)


def test_bits_wrapper_with_boolean_results(isolated_registry):
    ASTNode.reset()
    a, b = var_bit("a"), var_bit("b")
    assert_phase_oracle(bits_uint((a, ~b)) != const_uint(2))


CASES_TEST_GE = [{"a": 0, "b": 0}, {"a": 0, "b": 1}, {"a": 3, "b": 5}, {"a": 5, "b": 3}, {"a": 4, "b": 4}, {"a": 7, "b": 0}, {"a": 0, "b": 7}, {"a": 3, "b": 4}, {"a": 4, "b": 3}]


def test_ge(isolated_registry):
    ASTNode.reset()
    a, b = var_uint("a", 3), var_uint("b", 3)
    assert_phase_oracle(a >= b, lambda v: v["a"] >= v["b"], cases=CASES_TEST_GE)


CASES_TEST_GE_MISMATCHED_WIDTHS = [{"a": 0, "b": 3}, {"a": 3, "b": 3}, {"a": 4, "b": 3}, {"a": 2, "b": 3}, {"a": 7, "b": 0}, {"a": 1, "b": 2}, {"a": 4, "b": 0}]


def test_ge_mismatched_widths(isolated_registry):
    ASTNode.reset()
    a, b = var_uint("a", 3), var_uint("b", 2)
    assert_phase_oracle(a >= b, lambda v: v["a"] >= v["b"], cases=CASES_TEST_GE_MISMATCHED_WIDTHS)


CASES_TEST_GE_CONST = [{"a": 0}, {"a": 4}, {"a": 5}, {"a": 6}, {"a": 7}, {"a": 3}, {"a": 1}]


def test_ge_const(isolated_registry):
    ASTNode.reset()
    a = var_uint("a", 3)
    assert_phase_oracle(a >= const_uint(5, 3), lambda v: v["a"] >= 5, cases=CASES_TEST_GE_CONST)


CASES_TEST_OPTIMIZED_LAYOUT = [{"x": 0}, {"x": 1}, {"x": 2}, {"x": 3}]


def test_optimization_preserves_input_layout(isolated_registry):
    ASTNode.reset()
    x = var_uint("x", 2)
    oracle = assert_phase_oracle(x == x, cases=CASES_TEST_OPTIMIZED_LAYOUT)
    assert "x" in oracle.inputs
    assert len(oracle.inputs["x"]) == 2
