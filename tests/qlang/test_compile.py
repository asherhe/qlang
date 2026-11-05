"""Phase-oracle behavior, metadata, and validation regression tests."""

import itertools
import unittest
from unittest.mock import patch

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

    def test_uint_root_fails_before_circuit_creation(self):
        expr = var_uint("x", 2)
        with patch("qlang.compile.QuantumCircuit") as circuit:
            with self.assertRaisesRegex(TypeError, "expects a Bit"):
                compile_oracle(expr)  # pyright: ignore[reportArgumentType]
            circuit.assert_not_called()

    def test_invalid_nodes_and_operand_counts_are_rejected(self):
        a = var_bit("a")
        parent = var_uint("parent", 2)
        # Start with a valid variable, then break its structure so the compiler
        # performs the rejection instead of the expression constructor.
        variable_with_operand = var_bit("bad_operands")
        variable_with_operand.operands = (a,)
        cases = (
            Bit(bit_type=BitType.NOT),
            Bit(bit_type=BitType.AND, operands=(a,)),
            variable_with_operand,
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
        with patch("qlang.compile.QuantumCircuit") as circuit:
            for expr in cases:
                with self.subTest(node_id=id(expr)):
                    with self.assertRaises((TypeError, ValueError)):
                        compile_oracle(expr)
            circuit.assert_not_called()

    def test_cycles_are_rejected(self):
        cycle = Bit(bit_type=BitType.NOT)
        cycle.operands = (cycle,)
        with self.assertRaisesRegex(ValueError, "Cycle"):
            compile_oracle(cycle)

    def test_invalid_uint_metadata_is_rejected_before_allocation(self):
        x = var_uint("x", 2)
        root = x != 1
        # Mutate one field at a time and restore it before the next case.
        cases = (
            ("name", ""),
            ("name", " "),
            ("width", 0),
            ("width", -1),
            ("width", True),
            ("width", 1.5),
            ("uint_type", BitType.VAR),
            ("bits", list(x.bits)),
            ("bits", x.bits[:1]),
            ("bits", (const_bit(False), const_bit(False))),
            ("operands", []),
        )
        with patch("qlang.compile.QuantumCircuit") as circuit:
            for field, value in cases:
                with self.subTest(field=field, value=value):
                    original = getattr(x, field)
                    try:
                        setattr(x, field, value)
                        with self.assertRaises((TypeError, ValueError)):
                            compile_oracle(root)
                    finally:
                        setattr(x, field, original)
            circuit.assert_not_called()

    def test_owned_uint_bit_metadata_is_checked(self):
        x = var_uint("x", 2)
        y = var_uint("y", 2)
        root = x != 1
        cases = (
            ("index", True),
            ("index", 1),
            ("parent_uint", y),
            ("operands", (const_bit(False),)),
        )
        for field, value in cases:
            with self.subTest(field=field):
                original = getattr(x.bits[0], field)
                try:
                    setattr(x.bits[0], field, value)
                    with self.assertRaises(ValueError):
                        compile_oracle(root)
                finally:
                    setattr(x.bits[0], field, original)

    def test_indexed_bit_parent_is_validated(self):
        x = var_uint("x", 2)
        # The selected bit is valid, but a different owned bit has bad metadata.
        x.bits[0].index = -1
        with self.assertRaisesRegex(ValueError, "indices"):
            compile_oracle(x.bits[1])

    def test_constant_uint_requires_constant_bits(self):
        a = var_bit("a")
        malformed = UInt(uint_type=UIntType.CONST, width=1, bits=(a,))
        with self.assertRaisesRegex(TypeError, "constant Bit"):
            compile_oracle(malformed != 0)

    def test_constant_uint_bit_values_are_checked(self):
        bad_bit = Bit(bit_type=BitType.CONST, value=1)  # pyright: ignore[reportArgumentType]
        malformed = UInt(uint_type=UIntType.CONST, width=1, bits=(bad_bit,))
        with self.assertRaisesRegex(TypeError, "true or false"):
            compile_oracle(malformed != 0)

    def test_indexed_bit_must_belong_to_its_parent(self):
        x = var_uint("x", 2)
        extra_bit = Bit(bit_type=BitType.UINT, index=0, parent_uint=x)
        with self.assertRaisesRegex(ValueError, "stored at its position"):
            compile_oracle(extra_bit)

    def test_bits_wrapper_children_are_validated_before_allocation(self):
        wrapper = bits_uint((const_bit(False),))
        root = wrapper != 1
        invalid_children = (
            (42,),
            (Bit(bit_type=BitType.NOT),),
            (Bit(bit_type=BitType.CONST, value=1),),  # pyright: ignore[reportArgumentType]
        )
        with patch("qlang.compile.QuantumCircuit") as circuit:
            for children in invalid_children:
                with self.subTest(children=children):
                    wrapper.bits = children  # pyright: ignore[reportAttributeAccessIssue]
                    with self.assertRaises((TypeError, ValueError)):
                        compile_oracle(root)
            circuit.assert_not_called()

    def test_cycles_through_bits_wrapper_are_rejected(self):
        negated = ~var_bit("a")
        wrapper = bits_uint((negated,))
        root = wrapper != 0
        negated.operands = (root,)
        with self.assertRaisesRegex(ValueError, "Cycle"):
            compile_oracle(root)

    def test_invalid_operand_types_are_rejected(self):
        a, x = var_bit("a"), var_uint("x", 1)
        malformed = (
            Bit(bit_type=BitType.NOT, operands=(x,)),
            Bit(bit_type=BitType.LT, operands=(a, x)),
            UInt(uint_type=UIntType.ADD, width=2, operands=(x, a)),
        )
        for node in malformed:
            expr = node if isinstance(node, Bit) else node != 0
            with self.subTest(node_id=id(node)):
                with self.assertRaises(TypeError):
                    compile_oracle(expr)
