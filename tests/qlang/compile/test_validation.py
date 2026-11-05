"""Validation regression tests: malformed expressions are rejected before compilation.

Malformed-node cases deliberately use constructors or mutate fields to exercise
the compiler boundary independently of the factories. Each rejection case patches
``qlang.compile.compile.QuantumCircuit`` to prove that rejection happens before
any circuit is allocated.
"""

import unittest
from unittest.mock import patch

from qlang.compile import compile_oracle
from qlang.compile.compile import validate_expression
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


class TestValidation(unittest.TestCase):
    def setUp(self):
        circuit_patch = patch("qlang.compile.compile.QuantumCircuit")
        circuit = circuit_patch.start()
        self.addCleanup(circuit_patch.stop)
        self.addCleanup(circuit.assert_not_called)

    def test_uint_root_fails_before_circuit_creation(self):
        expr = var_uint("x", 2)
        with self.assertRaisesRegex(TypeError, "expects a Bit"):
            compile_oracle(expr)  # pyright: ignore[reportArgumentType]

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
        for expr in cases:
            with self.subTest(node_id=id(expr)):
                with self.assertRaises((TypeError, ValueError)):
                    compile_oracle(expr)

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
        for field, value in cases:
            with self.subTest(field=field, value=value):
                original = getattr(x, field)
                try:
                    setattr(x, field, value)
                    with self.assertRaises((TypeError, ValueError)):
                        compile_oracle(root)
                finally:
                    setattr(x, field, original)

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
        for children in invalid_children:
            with self.subTest(children=children):
                wrapper.bits = children  # pyright: ignore[reportAttributeAccessIssue]
                with self.assertRaises((TypeError, ValueError)):
                    compile_oracle(root)

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


class TestValidExpressions(unittest.TestCase):
    def test_shared_graphs_and_uint_back_references_are_not_cycles(self):
        a, b = var_bit("a"), var_bit("b")
        x, y = var_uint("x", 2), var_uint("y", 3)
        shared = a & b
        wrapper = bits_uint((shared, x.bits[0], const_bit(False)))
        expressions = (
            const_bit(True),
            shared ^ (shared | a),
            a == b,
            a < b,
            x.bits[1],
            x == const_uint(1, 2),
            (x + y) != 0,
            wrapper == 1,
            bits_uint((x.bits[0], x.bits[0])) != 0,
        )
        for expr in expressions:
            with self.subTest(node_id=id(expr)):
                validate_expression(expr)


if __name__ == "__main__":
    unittest.main()
