"""Phase-oracle behavior and metadata regression tests."""

import itertools
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlang.compile import Oracle, OutputMode, compile_oracle
from qlang.compile.compile import parse_input
from qlang.expression import (
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
from qlang.evaluate import evaluate


class TestCompile(unittest.TestCase):
    def assert_phase_oracle(self, expr):
        """Compare circuit phases with evaluation of the original expression."""
        # Gather the original variables before optimization can remove them.
        # The shared pytest fixture resets both the cache and variable registry.
        reference_widths = {}
        for node in parse_input(expr):
            if isinstance(node, UInt) and node.uint_type == UIntType.VAR:
                reference_widths[node.name] = node.width
            elif isinstance(node, Bit) and node.bit_type == BitType.VAR:
                reference_widths[node.name] = 1

        oracle = compile_oracle(expr)
        self.assertIsInstance(oracle, Oracle)
        self.assertEqual(oracle.output, [])
        self.assertEqual(oracle.dirty_ancillas, [])
        input_qubits = [qubit for register in oracle.inputs.values() for qubit in register]
        self.assertEqual(len(input_qubits), len(set(input_qubits)))
        self.assertEqual(len(oracle.clean_ancillas), len(set(oracle.clean_ancillas)))
        self.assertFalse(set(input_qubits) & set(oracle.clean_ancillas))
        self.assertEqual(set(input_qubits) | set(oracle.clean_ancillas), set(range(oracle.circuit.num_qubits)))

        for name, register in oracle.inputs.items():
            self.assertEqual(len(register), reference_widths[name])

        names = list(reference_widths)
        domains = [range(1 << reference_widths[name]) for name in names]
        expected_superposition = np.zeros(2 ** oracle.circuit.num_qubits, dtype=complex)
        for values in itertools.product(*domains):
            assignments = dict(zip(names, values))
            basis = sum(
                ((assignments[name] >> bit) & 1) << qubit
                for name, register in oracle.inputs.items()
                for bit, qubit in enumerate(register)
            )
            sign = -1 if evaluate(expr, assignments) else 1
            expected = np.zeros_like(expected_superposition)
            expected[basis] = sign
            actual = Statevector.from_int(basis, len(expected)).evolve(oracle.circuit)
            # Exact positions and signs detect input changes, leftover scratch,
            # and wrong global signs on constant predicates as well.
            np.testing.assert_allclose(actual.data, expected, atol=1e-12)
            expected_superposition[basis] = sign / np.sqrt(2 ** len(input_qubits))

        prepared = QuantumCircuit(oracle.circuit.num_qubits)
        for qubit in input_qubits:
            prepared.h(qubit)
        prepared.compose(oracle.circuit, inplace=True)
        np.testing.assert_allclose(Statevector.from_instruction(prepared).data, expected_superposition, atol=1e-12)
        return oracle

    def test_oracle_metadata_and_ancilla_count(self):
        circuit = QuantumCircuit(5)
        oracle = Oracle(circuit, {"x": [0, 1]}, [(2, OutputMode.BIT_FLIP)], [3], [4])
        self.assertIs(oracle.circuit, circuit)
        self.assertEqual(oracle.inputs, {"x": [0, 1]})
        self.assertEqual(oracle.output, [(2, OutputMode.BIT_FLIP)])
        self.assertEqual(oracle.total_ancillas, 2)

    def test_ancilla_defaults_are_independent(self):
        first, second = Oracle(QuantumCircuit(1), {}, []), Oracle(QuantumCircuit(1), {}, [])
        first.clean_ancillas.append(0)
        first.dirty_ancillas.append(0)
        self.assertEqual(second.clean_ancillas, [])
        self.assertEqual(second.dirty_ancillas, [])
        self.assertEqual(second.total_ancillas, 0)

    def test_output_modes_are_distinct(self):
        self.assertIsNot(OutputMode.BIT_FLIP, OutputMode.PHASE)

    def test_compile_returns_oracle(self):
        self.assertIsInstance(compile_oracle(var_bit("p")), Oracle)

    def test_boolean_operations_and_shared_results(self):
        a, b, c = [var_bit(name) for name in "abc"]
        shared = a | b
        for expr in (a & b, a | b, a ^ b, ~a, (a | b) & ~c, shared ^ (shared & c)):
            with self.subTest(operation=expr.bit_type.name):
                self.assert_phase_oracle(expr)

    def test_repeated_operands(self):
        a = var_bit("a")
        for expr in (a & a, a | a, a ^ a):
            with self.subTest(operation=expr.bit_type.name):
                self.assert_phase_oracle(expr)

    def test_bit_comparisons(self):
        a, b = var_bit("a"), var_bit("b")
        for expr in (a != b, a < b):
            with self.subTest(operation=expr.bit_type.name):
                self.assert_phase_oracle(expr)

    def test_uint_comparisons_with_unequal_widths(self):
        x, y = var_uint("x", 1), var_uint("y", 2)
        # Cover padding on both sides of < and reuse of the same input.
        for expr in (x != y, x < y, y < x, y != y, y < y):
            with self.subTest(expression=expr):
                self.assert_phase_oracle(expr)

    def test_uint_comparison_inside_boolean_expression(self):
        x, enabled = var_uint("x", 2), var_bit("enabled")
        self.assert_phase_oracle((x < 2) & enabled)

    def test_leaf_roots_and_constant_preparation(self):
        a, b = var_bit("a"), var_bit("b")
        true, false = const_bit(True), const_bit(False)
        for expr in (a, true, false, a & true, a & false, (a & true) | (b & true)):
            with self.subTest(operation=expr.bit_type.name):
                self.assert_phase_oracle(expr)

    def test_indexed_inputs_are_grouped_in_bit_order(self):
        x = var_uint("x", 3)
        oracle = self.assert_phase_oracle(x.bits[2] & x.bits[0])
        self.assertEqual(len(oracle.inputs["x"]), 3)
        self.assertEqual(len(oracle.clean_ancillas), 1)

    def test_single_indexed_bit_preserves_full_parent_width(self):
        x = var_uint("x", 3)
        oracle = self.assert_phase_oracle(x.bits[1])
        self.assertEqual(len(oracle.inputs["x"]), 3)
        self.assertEqual(oracle.clean_ancillas, [])

    def test_uint_equality_is_compiled_through_optimization(self):
        self.assert_phase_oracle(var_uint("x", 2) == 2)

    def test_addition_predicate(self):
        x, y = var_uint("x", 1), var_uint("y", 1)
        self.assert_phase_oracle(x + y == 2)

    def test_bits_wrapper_with_boolean_results(self):
        a, b = var_bit("a"), var_bit("b")
        self.assert_phase_oracle(bits_uint((a, ~b)) != const_uint(2))

    def test_addition_with_shared_input_bits(self):
        x = var_uint("x", 2)
        self.assert_phase_oracle(x + bits_uint(x.bits) == 2)

    def test_optimization_preserves_input_layout(self):
        x = var_uint("x", 2)
        oracle = self.assert_phase_oracle(x == x)
        self.assertIn("x", oracle.inputs)
        self.assertEqual(len(oracle.inputs["x"]), 2)

