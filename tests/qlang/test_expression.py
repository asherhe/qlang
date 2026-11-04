"""Factories, symbolic operators, widths, and structural sharing."""

import operator

import pytest

from qlang.expression import Bit, BitType, UInt, UIntType, bits_uint, const_bit, const_uint, uint, var_bit, var_uint


@pytest.mark.parametrize("value,expected", [(0, False), (1, True), (2, True), (False, False), (True, True)])
def test_constant_bit_conversion(value, expected):
    assert const_bit(value).value is expected


@pytest.mark.parametrize("value,width", [(0, 1), (1, 1), (6, 3), (15, 4)])
def test_inferred_constant_width_and_little_endian_bits(value, width):
    node = uint(value)
    assert node.width == width
    assert node.const_value == value
    assert tuple(b.value for b in node.bits) == tuple(bool(value & (1 << i)) for i in range(width))


def test_explicit_constant_width():
    node = const_uint(3, 5)
    assert node.width == 5
    assert node.const_value == 3


@pytest.mark.parametrize("value,width", [(-1, None), (8, 3), (0, 0), (0, -1)])
def test_invalid_constants_rejected(value, width):
    with pytest.raises(ValueError):
        const_uint(value, width)


def test_public_variable_factory_supports_bit_zero():
    node = var_uint("x", 3)
    assert node.name == "x"
    assert node.width == 3
    assert [b.index for b in node.bits] == [0, 1, 2]
    assert all(b.name == "x" and b.parent_width == 3 for b in node.bits)


def test_uint_dispatches_variable_factory():
    node = uint("x", 2)
    assert isinstance(node, UInt)
    assert node.uint_type == UIntType.VAR


@pytest.mark.parametrize("width", [0, -1])
def test_invalid_variable_width_rejected(width):
    with pytest.raises(ValueError):
        var_uint("x", width)


def test_variable_requires_width():
    with pytest.raises(ValueError):
        uint("x")


def test_index_zero_is_valid():
    bit = var_bit("x", index=0, parent_width=3)
    assert bit.index == 0


@pytest.mark.parametrize("index,width", [(-1, 3), (3, 3), (1, -1), (1, 0)])
def test_invalid_index_metadata_rejected(index, width):
    with pytest.raises(ValueError):
        var_bit("x", index=index, parent_width=width)


@pytest.mark.parametrize("op,kind,width", [(operator.add, UIntType.ADD, 4), (operator.sub, UIntType.SUB, 3), (operator.mul, UIntType.MUL, 5), (operator.lshift, UIntType.LSHIFT, 3), (operator.rshift, UIntType.RSHIFT, 3)])
def test_arithmetic_nodes_and_widths(x, y, op, kind, width):
    node = op(x, y)
    assert node.uint_type == kind
    assert node.width == width
    assert node.operands[0] is x and node.operands[1] is y


@pytest.mark.parametrize("op,kind", [(operator.eq, BitType.EQ), (operator.ne, BitType.NE), (operator.lt, BitType.LT), (operator.le, BitType.LE), (operator.gt, BitType.GT), (operator.ge, BitType.GE)])
def test_comparisons_build_predicates(x, y, op, kind):
    node = op(x, y)
    assert isinstance(node, Bit)
    assert node.bit_type == kind


def test_reflected_noncommutative_operands(x):
    for node, kind in [(1 - x, UIntType.SUB), (1 << x, UIntType.LSHIFT), (8 >> x, UIntType.RSHIFT)]:
        assert node.uint_type == kind
        assert node.operands[1] is x
        assert node.operands[0].uint_type == UIntType.CONST


def test_symbolic_truth_testing_is_forbidden(x, p):
    for node in (x, p, const_bit(False), x == 1):
        with pytest.raises(TypeError, match="booleans"):
            bool(node)


def test_operation_nodes_are_shared_by_operand_identity(x, y, p, q):
    assert (x + y) is (x + y)
    assert (p & q) is (p & q)
    assert (x - y) is not (y - x)


def test_uint_cache_key_includes_width(x):
    amount = const_uint(1)
    narrow = UInt.cache_int(UIntType.LSHIFT, 3, (x, amount))
    wide = UInt.cache_int(UIntType.LSHIFT, 4, (x, amount))
    assert narrow is not wide
    assert (narrow.width, wide.width) == (3, 4)


def test_leaf_property_and_variable_const_value(x):
    assert x.is_leaf
    assert not (x + 1).is_leaf
    with pytest.raises(ValueError):
        _ = x.const_value


# -------- BitType.UINT / UIntType.BITS --------
# These build nodes with the public factories and only inspect structure; evaluate() has no
# BITS/UINT support yet.

def test_new_enum_members_are_distinct_from_existing_ones():
    assert BitType.UINT not in (BitType.CONST, BitType.VAR)
    assert UIntType.BITS not in (UIntType.CONST, UIntType.VAR)
    assert len(set(BitType)) == len(BitType)
    assert len(set(UIntType)) == len(UIntType)


def test_var_uint_creates_one_uint_bit_per_position():
    node = var_uint("x", 4)
    assert len(node.bits) == 4
    for i, bit in enumerate(node.bits):
        assert bit.bit_type == BitType.UINT
        assert bit.index == i
        assert bit.parent_uint is node
        assert bit.is_leaf


def test_uint_bits_are_shared_nodes():
    node = var_uint("x", 3)
    assert Bit._create(bit_type=BitType.UINT, index=1, parent_uint=node) is node.bits[1]


def test_uint_bits_of_different_parents_are_distinct():
    assert var_uint("x", 3).bits[0] is not var_uint("y", 3).bits[0]


def test_operator_uint_also_owns_its_bits():
    total = var_uint("x", 3) + var_uint("y", 2)
    assert len(total.bits) == total.width == 4
    assert all(b.bit_type == BitType.UINT and b.parent_uint is total for b in total.bits)
    assert [b.index for b in total.bits] == [0, 1, 2, 3]


def test_const_uint_keeps_const_bits():
    node = const_uint(5, 4)
    assert all(b.bit_type == BitType.CONST and b.parent_uint is None for b in node.bits)


def test_var_bit_is_standalone():
    bit = var_bit("p")
    assert bit.bit_type == BitType.VAR
    assert bit.name == "p"
    assert bit.index is None
    assert bit.parent_uint is None


def test_bits_uint_builds_bits_node():
    bits = (const_bit(1), var_bit("p"), const_bit(0))
    node = bits_uint(bits)
    assert isinstance(node, UInt)
    assert node.uint_type == UIntType.BITS
    assert node.width == 3
    assert node.bits == bits
    assert node.is_leaf


def test_bits_uint_accepts_bits_owned_by_other_uints():
    node_x = var_uint("x", 3)
    node = bits_uint((node_x.bits[2], node_x.bits[0]))
    assert node.width == 2
    assert node.bits[0] is node_x.bits[2] and node.bits[1] is node_x.bits[0]


def test_bits_uint_works_as_operand():
    node = bits_uint((const_bit(1), const_bit(0), const_bit(1)))
    total = node + 1
    assert total.uint_type == UIntType.ADD
    assert total.operands[0] is node
    assert total.width == 4


def test_uint_dispatches_tuple_to_bits_uint():
    bits = (const_bit(1), const_bit(1))
    node = uint(bits)
    assert node.uint_type == UIntType.BITS
    assert node.bits == bits


@pytest.mark.parametrize("value", [1.5, None, [const_bit(1)]])
def test_uint_rejects_unsupported_types(value):
    with pytest.raises(TypeError):
        uint(value)


@pytest.mark.xfail(
    strict=True,
    reason="BITS nodes are cached by (type, width, operands) and have no operands, so same-width BITS UInts collide",
)
def test_bits_uint_with_different_bits_are_distinct_nodes():
    first = bits_uint((const_bit(1), const_bit(0)))
    second = bits_uint((var_bit("p"), var_bit("q")))
    assert first is not second
    assert second.bits[0].name == "p"
