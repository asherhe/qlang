"""Shared fixtures; keep evaluator tests independent of public factory failures."""

import pytest

from qlang.expression import ASTNode, UInt, UIntType, var_bit


@pytest.fixture(autouse=True)
def isolate_expression_cache():
    ASTNode.reset()
    yield


@pytest.fixture
def make_uint():
    # Construct valid AST inputs directly so a factory regression does not hide
    # independent evaluator and optimizer failures. Public factories have tests.
    def make(name="x", width=3):
        # the UInt constructor creates the BitType.UINT bits owned by the node
        return UInt(uint_type=UIntType.VAR, name=name, width=width)
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
