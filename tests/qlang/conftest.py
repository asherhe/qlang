"""Shared fixtures; keep evaluator tests independent of public factory failures."""

import pytest

from qlang.expression import ASTNode, Bit, BitType, UInt, UIntType, var_bit


@pytest.fixture(autouse=True)
def isolate_expression_cache():
    saved = ASTNode.cache
    ASTNode.cache = {}
    yield
    ASTNode.cache = saved


@pytest.fixture
def make_uint():
    # Construct valid AST inputs directly so a factory regression does not hide
    # independent evaluator and optimizer failures. Public factories have tests.
    def make(name="x", width=3):
        bits = tuple(
            Bit(bit_type=BitType.VAR, name=name, index=i, parent_width=width)
            for i in range(width)
        )
        return UInt(uint_type=UIntType.VAR, name=name, width=width, bits=bits)
    return make


@pytest.fixture
def x(make_uint):
    return make_uint("x", 3)


@pytest.fixture
def y(make_uint):
    return make_uint("y", 2)


@pytest.fixture
def p():
    return var_bit("p")


@pytest.fixture
def q():
    return var_bit("q")
