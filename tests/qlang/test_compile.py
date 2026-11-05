"""Phase-oracle behavior, metadata, and validation regression tests."""

import itertools
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qlang.compile import Oracle, OutputMode, compile_oracle
from qlang.expression import ASTNode, Bit, BitType, UInt, UIntType, const_bit, var_bit, var_uint


class TestCompile(unittest.TestCase):
    def setUp(self):
        self.saved_cache = ASTNode.cache
        ASTNode.cache = {}

    def tearDown(self):
        ASTNode.cache = self.saved_cache

    def assert_phase_oracle(self, expr):
        """Compare the public result with an independent Boolean evaluator."""
        oracle = compile_oracle(expr)
        self.assertIsInstance(oracle, Oracle)
        self.assertEqual(oracle.output, [])
        self.assertEqual(oracle.dirty_ancillas, [])
        input_qubits = [qubit for register in oracle.inputs.values() for qubit in register]
        self.assertEqual(len(input_qubits), len(set(input_qubits)))
        self.assertEqual(len(oracle.clean_ancillas), len(set(oracle.clean_ancillas)))
        self.assertFalse(set(input_qubits) & set(oracle.clean_ancillas))
        self.assertEqual(set(input_qubits) | set(oracle.clean_ancillas), set(range(oracle.circuit.num_qubits)))

        def evaluate(node, assignments):
            if isinstance(node, UInt):
                if node.uint_type == UIntType.VAR:
                    return assignments[node.name]
                if node.uint_type in (UIntType.CONST, UIntType.BITS):
                    return sum(
                        int(evaluate(bit, assignments)) << index
                        for index, bit in enumerate(node.bits)
                    )
                raise AssertionError(f"Unexpected reference operation: {node.uint_type}")

            kind = node.bit_type
            if kind == BitType.VAR:
                return bool(assignments[node.name])
            if kind == BitType.UINT:
                # Indexed leaves read one bit of the parent's integer assignment.
                return bool((assignments[node.parent_uint.name] >> node.index) & 1)
            if kind == BitType.CONST:
                return node.value
            values = [evaluate(child, assignments) for child in node.operands]
            if kind == BitType.NOT:
                return not values[0]
            if kind == BitType.AND:
                return values[0] and values[1]
            if kind == BitType.OR:
                return values[0] or values[1]
            if kind in (BitType.XOR, BitType.NE):
                return values[0] != values[1]
            if kind == BitType.LT:
                return values[0] < values[1]
            raise AssertionError(f"Unexpected reference operation: {kind}")

        names = list(oracle.inputs)
        domains = [range(1 << len(oracle.inputs[name])) for name in names]
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

    def test_distinct_objects_naming_same_input_share_a_qubit(self):
        a = Bit(bit_type=BitType.VAR, name="a")
        another_a = Bit(bit_type=BitType.VAR, name="a")
        oracle = self.assert_phase_oracle(a ^ another_a)
        self.assertEqual(oracle.circuit.num_qubits, 2)

    def test_unsupported_operations_fail_before_circuit_creation(self):
        a, b = var_bit("a"), var_bit("b")
        with patch("qlang.compile.QuantumCircuit") as circuit:
            for expr in (a == b, a & (a == b), var_uint("x", 2)):
                with self.subTest(node=type(expr).__name__):
                    with self.assertRaises((TypeError, ValueError)):
                        compile_oracle(expr)
            circuit.assert_not_called()

    def test_invalid_nodes_and_operand_counts_are_rejected(self):
        a = var_bit("a")
        parent = var_uint("parent", 2)
        # Pass malformed arguments on purpose to exercise runtime validation.
        cases = (
            Bit(bit_type=BitType.NOT),
            Bit(bit_type=BitType.AND, operands=(a,)),
            Bit(bit_type=BitType.VAR, name="a", operands=(a,)),
            Bit(bit_type=BitType.NOT, operands=[a]),  # pyright: ignore[reportArgumentType]
            Bit(bit_type=BitType.NOT, operands=(42,)),  # pyright: ignore[reportArgumentType]
            Bit(bit_type=UIntType.VAR, name="bad"),  # pyright: ignore[reportArgumentType]
            Bit(bit_type=BitType.VAR, name=" "),
            Bit(bit_type=BitType.CONST, value=1),  # pyright: ignore[reportArgumentType]
            Bit(bit_type=BitType.UINT, index=-1, parent_uint=parent),
            Bit(bit_type=BitType.UINT, index=True, parent_uint=parent),
            Bit(bit_type=BitType.UINT, index=2, parent_uint=parent),
            Bit(bit_type=BitType.UINT, index=0),
        )
        for expr in cases:
            with self.subTest(node_id=id(expr)):
                with self.assertRaises((TypeError, ValueError)):
                    compile_oracle(expr)

    def test_mixed_comparison_operands_are_rejected(self):
        a, x = var_bit("a"), var_uint("x", 2)
        expr = Bit(bit_type=BitType.LT, operands=(a, x))
        with self.assertRaisesRegex(TypeError, "must both be Bit or both be UInt"):
            compile_oracle(expr)

    def test_cycles_are_rejected(self):
        cycle = Bit(bit_type=BitType.NOT)
        cycle.operands = (cycle,)
        with self.assertRaisesRegex(ValueError, "Cycle"):
            compile_oracle(cycle)

    def test_conflicting_input_definitions_are_rejected(self):
        cases = (
            var_bit("x") & var_uint("x", 2).bits[0],
            var_uint("x", 2).bits[0] & var_uint("x", 3).bits[1],
        )
        for expr in cases:
            with self.subTest(node_id=id(expr)):
                with self.assertRaises(ValueError):
                    compile_oracle(expr)
