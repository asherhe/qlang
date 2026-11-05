"""Gadget execution tests: each @gadget-dispatched operator compiles to a valid phase oracle."""

import itertools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlang.compile.compile import compile_oracle
from qlang.compile.gadgets import _GADGETS
from qlang.expression import ASTNode, const_bit, var_bit, var_uint, Bit, BitType
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
    """Compare the public result with an independent Boolean evaluator."""
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

    for values in itertools.product(*domains):
        assignments = dict(zip(names, values))
        basis = sum(
            ((assignments[name] >> bit) & 1) << qubit
            for name, register in oracle.inputs.items()
            for bit, qubit in enumerate(register)
        )
        sign = -1 if mock_evaluate(expr, assignments) else 1
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
