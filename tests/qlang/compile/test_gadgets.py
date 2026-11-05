"""Gadget execution tests: each @gadget-dispatched operator compiles to a valid phase oracle."""

import itertools

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qlang.compile.compile import compile_oracle
from qlang.compile.gadgets import _GADGETS
from qlang.expression import ASTNode, const_bit, var_bit, var_uint
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


def assert_phase_oracle(expr):
    """independent check: the circuit applies -1 iff the expression is true"""
    oracle = compile_oracle(expr)
    names = list(oracle.inputs)
    domains = [range(1 << len(oracle.inputs[n])) for n in names]
    expected = np.zeros(2 ** oracle.circuit.num_qubits, dtype=complex)
    for values in itertools.product(*domains):
        assignments = dict(zip(names, values))
        basis = sum(
            ((assignments[name] >> bit) & 1) << qubit
            for name, register in oracle.inputs.items()
            for bit, qubit in enumerate(register)
        )
        sign = -1 if evaluate(expr, assignments) else 1
        expected[basis] = sign
        actual = Statevector.from_int(basis, oracle.circuit.num_qubits).evolve(oracle.circuit)
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


def test_uint_add(isolated_registry):
    ASTNode.reset()
    x, y = var_uint("x", 2), var_uint("y", 2)
    assert_phase_oracle(x + y)


def test_indexed_uint_bit(isolated_registry):
    ASTNode.reset()
    x = var_uint("x", 3)
    oracle = assert_phase_oracle(x.bits[2] & x.bits[0])
    assert len(oracle.inputs["x"]) == 3
    assert len(oracle.clean_ancillas) == 1