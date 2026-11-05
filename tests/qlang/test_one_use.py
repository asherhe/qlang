"""Use-count analysis that annotates qubits with `one_use`."""

from qiskit.circuit import Qubit

from qlang.compile.compile import count_uses, is_one_use, parse_input
from qlang.compile.qubit import Q


def uses_of(expr):
    return count_uses(parse_input(expr), expr)


def test_root_is_used_once(p, q):
    expr = p ^ q
    uses = uses_of(expr)
    assert is_one_use(uses, expr)


def test_single_consumer_is_one_use(p, q):
    uses = uses_of(p ^ q)
    assert is_one_use(uses, p)
    assert is_one_use(uses, q)


def test_repeated_operand_is_not_one_use(p):
    uses = uses_of(p & p)
    assert not is_one_use(uses, p)


def test_shared_subexpression_is_not_one_use(p, q):
    shared = p & q
    expr = shared ^ (shared | p)
    uses = uses_of(expr)
    assert not is_one_use(uses, shared)
    # p is consumed by `p & q` and by `shared | p`
    assert not is_one_use(uses, p)
    assert is_one_use(uses, q)


def test_uint_bits_follow_their_consumers(x, y):
    total = x + y
    uses = uses_of(total.bits[0] ^ total.bits[1])
    assert all(is_one_use(uses, b) for b in x.bits)
    assert all(is_one_use(uses, b) for b in y.bits)


def test_uint_used_twice_is_not_one_use(x, y):
    total = x + y
    uses = uses_of((total + total).bits[0])
    assert all(not is_one_use(uses, b) for b in total.bits)
    assert all(is_one_use(uses, b) for b in x.bits)


def test_one_use_does_not_affect_qubit_equality():
    raw = Qubit()
    assert Q(raw, one_use=True) == Q(raw, one_use=False)
