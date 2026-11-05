"""Gadget execution tests: each @gadget-dispatched operator compiles to a valid phase oracle."""

import itertools
from collections import defaultdict

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlang.compile.compile import compile_oracle
from qlang.compile.gadgets import _GADGETS
from qlang.expression import ASTNode, const_bit, const_uint, var_bit, var_uint, Bit, BitType
from qlang.evaluate import evaluate


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
    """Compare the public result with an independent evaluator.

    expected_fn, if given, is a plain-python function of the input assignments returning the
    predicate's truth value. inputs the optimizer removed read as 0, since they cannot matter.    """
    oracle = compile_oracle(expr)
    input_qubits = [q for register in oracle.inputs.values() for q in register]
    names = list(oracle.inputs)
    domains = [range(1 << len(oracle.inputs[name])) for name in names]
    expected_superposition = np.zeros(2 ** oracle.circuit.num_qubits, dtype=complex)

    def mock_evaluate(node, assignments):
        """Evaluate a Bit node with side-effect-free recursion."""
        if isinstance(node, Bit) and node.bit_type == BitType.VAR:
            # variables the optimizer removed (e.g. a ^ a) are absent and can't affect the result
            return bool(assignments.get(node.name, 0))
        if isinstance(node, Bit) and node.bit_type == BitType.UINT:
            return bool((assignments[node.parent_uint.name] >> node.index) & 1)
        if isinstance(node, Bit) and node.bit_type == BitType.CONST:
            return bool(node.value)
        if isinstance(node, Bit) and node.bit_type == BitType.NOT:
            return not mock_evaluate(node.operands[0], assignments)
        if isinstance(node, Bit) and node.bit_type in (BitType.XOR, BitType.AND, BitType.OR):
            left = mock_evaluate(node.operands[0], assignments)
            right = mock_evaluate(node.operands[1], assignments)
            if node.bit_type == BitType.XOR:
                return left ^ right
            elif node.bit_type == BitType.AND:
                return left and right
            elif node.bit_type == BitType.OR:
                return left or right
        return False

    if cases is None:
        cases = [dict(zip(names, values)) for values in itertools.product(*domains)]

    for case in cases:
        assignments = {name: case[name] for name in names}
        basis = sum(
            ((assignments[name] >> bit) & 1) << qubit
            for name, register in oracle.inputs.items()
            for bit, qubit in enumerate(register)
        )
        if expected_fn is not None:
            truth = expected_fn(defaultdict(int, case))
        else:
            truth = mock_evaluate(expr, assignments)
        sign = -1 if truth else 1
        expected = np.zeros_like(expected_superposition)
        expected[basis] = sign

        # Prepare basis state
        prepared = QuantumCircuit(oracle.circuit.num_qubits)
        for q in input_qubits:
            if (basis >> q) & 1:
                prepared.x(q)
        prepared.compose(oracle.circuit, inplace=True)
        actual = Statevector.from_instruction(prepared)
        np.testing.assert_allclose(actual.data, expected, atol=1e-12)
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


def test_indexed_uint_bit(isolated_registry):
    ASTNode.reset()
    x = var_uint("x", 3)
    oracle = assert_phase_oracle(x.bits[2] & x.bits[0])
    assert len(oracle.inputs["x"]) == 3
    assert len(oracle.clean_ancillas) == 1


# ---- UInt gadgets: each is checked through a != comparison so the predicate is a single Bit ----


CASES_TEST_ADD = [{"a": 0, "b": 0, "c": 0}, {"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 3, "c": 6}, {"a": 3, "b": 1, "c": 4}, {"a": 2, "b": 1, "c": 5}, {"a": 3, "b": 3, "c": 5}, {"a": 1, "b": 1, "c": 3}]


def test_add(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 2), var_uint("b", 2), var_uint("c", 3)
    assert_phase_oracle((a + b) != c, lambda v: v["a"] + v["b"] != v["c"], cases=CASES_TEST_ADD)


CASES_TEST_ADD_MISMATCHED_WIDTHS = [{"a": 0, "b": 0, "c": 0}, {"a": 7, "b": 1, "c": 8}, {"a": 7, "b": 0, "c": 7}, {"a": 5, "b": 1, "c": 6}, {"a": 7, "b": 1, "c": 7}, {"a": 4, "b": 1, "c": 0}, {"a": 3, "b": 1, "c": 5}, {"a": 6, "b": 1, "c": 7}]


def test_add_mismatched_widths(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 1), var_uint("c", 4)
    assert_phase_oracle((a + b) != c, lambda v: v["a"] + v["b"] != v["c"], cases=CASES_TEST_ADD_MISMATCHED_WIDTHS)


CASES_TEST_ADD_CONST = [{"a": 0, "c": 3}, {"a": 1, "c": 4}, {"a": 2, "c": 5}, {"a": 3, "c": 6}, {"a": 3, "c": 5}, {"a": 0, "c": 0}, {"a": 1, "c": 3}]


def test_add_const(isolated_registry):
    ASTNode.reset()
    a, c = var_uint("a", 2), var_uint("c", 3)
    assert_phase_oracle((a + 3) != c, lambda v: v["a"] + 3 != v["c"], cases=CASES_TEST_ADD_CONST)


CASES_TEST_SUB = [{"a": 3, "b": 1, "c": 2}, {"a": 0, "b": 1, "c": 3}, {"a": 2, "b": 2, "c": 0}, {"a": 1, "b": 0, "c": 1}, {"a": 0, "b": 0, "c": 1}, {"a": 2, "b": 3, "c": 2}, {"a": 3, "b": 3, "c": 3}]


def test_sub(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 2), var_uint("b", 2), var_uint("c", 2)
    assert_phase_oracle((a - b) != c, lambda v: (v["a"] - v["b"]) % 4 != v["c"], cases=CASES_TEST_SUB)


CASES_TEST_SUB_MISMATCHED_WIDTHS = [{"a": 5, "b": 1, "c": 4}, {"a": 0, "b": 1, "c": 7}, {"a": 7, "b": 0, "c": 7}, {"a": 4, "b": 1, "c": 3}, {"a": 4, "b": 1, "c": 4}, {"a": 0, "b": 1, "c": 0}, {"a": 6, "b": 0, "c": 5}]


def test_sub_mismatched_widths(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 1), var_uint("c", 3)
    assert_phase_oracle((a - b) != c, lambda v: (v["a"] - v["b"]) % 8 != v["c"], cases=CASES_TEST_SUB_MISMATCHED_WIDTHS)


CASES_TEST_LSHIFT_CONST = [{"a": 0, "c": 0}, {"a": 3, "c": 6}, {"a": 5, "c": 2}, {"a": 7, "c": 6}, {"a": 4, "c": 0}, {"a": 7, "c": 7}, {"a": 1, "c": 1}]


def test_lshift_const(isolated_registry):
    ASTNode.reset()
    a, c = var_uint("a", 3), var_uint("c", 3)
    assert_phase_oracle((a << 1) != c, lambda v: (v["a"] << 1) % 8 != v["c"], cases=CASES_TEST_LSHIFT_CONST)


CASES_TEST_LSHIFT_VARIABLE = [{"a": 1, "b": 2, "c": 4}, {"a": 5, "b": 1, "c": 2}, {"a": 7, "b": 3, "c": 0}, {"a": 3, "b": 0, "c": 3}, {"a": 3, "b": 0, "c": 0}, {"a": 6, "b": 1, "c": 6}]


@pytest.mark.xfail(reason="shifting by a variable amount is not supported yet", raises=TypeError)
def test_lshift_variable(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 2), var_uint("c", 3)
    assert_phase_oracle((a << b) != c, lambda v: (v["a"] << v["b"]) % 8 != v["c"], cases=CASES_TEST_LSHIFT_VARIABLE)


CASES_TEST_RSHIFT_CONST = [{"a": 0, "c": 0}, {"a": 7, "c": 3}, {"a": 5, "c": 2}, {"a": 1, "c": 0}, {"a": 3, "c": 1}, {"a": 6, "c": 2}, {"a": 4, "c": 4}]


def test_rshift_const(isolated_registry):
    ASTNode.reset()
    a, c = var_uint("a", 3), var_uint("c", 3)
    assert_phase_oracle((a >> 1) != c, lambda v: (v["a"] >> 1) != v["c"], cases=CASES_TEST_RSHIFT_CONST)


CASES_TEST_RSHIFT_VARIABLE = [{"a": 6, "b": 1, "c": 3}, {"a": 7, "b": 3, "c": 0}, {"a": 5, "b": 0, "c": 5}, {"a": 4, "b": 2, "c": 1}, {"a": 4, "b": 2, "c": 2}, {"a": 7, "b": 1, "c": 7}]


@pytest.mark.xfail(reason="shifting by a variable amount is not supported yet", raises=TypeError)
def test_rshift_variable(isolated_registry):
    ASTNode.reset()
    a, b, c = var_uint("a", 3), var_uint("b", 2), var_uint("c", 3)
    assert_phase_oracle((a >> b) != c, lambda v: (v["a"] >> v["b"]) != v["c"], cases=CASES_TEST_RSHIFT_VARIABLE)


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


CASES_TEST_LT = [{"a": 0, "b": 0}, {"a": 0, "b": 1}, {"a": 3, "b": 5}, {"a": 5, "b": 3}, {"a": 4, "b": 4}, {"a": 7, "b": 0}, {"a": 0, "b": 7}, {"a": 3, "b": 4}, {"a": 4, "b": 3}]


def test_lt(isolated_registry):
    ASTNode.reset()
    a, b = var_uint("a", 3), var_uint("b", 3)
    assert_phase_oracle(a < b, lambda v: v["a"] < v["b"], cases=CASES_TEST_LT)


CASES_TEST_LT_MISMATCHED_WIDTHS = [{"a": 0, "b": 3}, {"a": 3, "b": 3}, {"a": 4, "b": 3}, {"a": 2, "b": 3}, {"a": 7, "b": 0}, {"a": 1, "b": 2}, {"a": 4, "b": 0}]


def test_lt_mismatched_widths(isolated_registry):
    ASTNode.reset()
    a, b = var_uint("a", 3), var_uint("b", 2)
    assert_phase_oracle(a < b, lambda v: v["a"] < v["b"], cases=CASES_TEST_LT_MISMATCHED_WIDTHS)


CASES_TEST_LT_CONST = [{"a": 0}, {"a": 4}, {"a": 5}, {"a": 6}, {"a": 7}, {"a": 3}, {"a": 1}]


def test_lt_const(isolated_registry):
    ASTNode.reset()
    a = var_uint("a", 3)
    assert_phase_oracle(a < const_uint(5, 3), lambda v: v["a"] < 5, cases=CASES_TEST_LT_CONST)
