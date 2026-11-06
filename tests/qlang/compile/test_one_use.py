"""Use-count analysis that annotates qubits with `one_use`."""

from qiskit.circuit import Qubit

from qlang.compile.compile import count_uses, parse_input
from qlang.compile.qubit import Q


def uses_of(expr):
    return count_uses(parse_input(expr), expr)


def one_use(uses, bit):
    return uses.get(id(bit), 0) == 1


def test_root_is_used_once(p, q):
    expr = p ^ q
    uses = uses_of(expr)
    assert one_use(uses, expr)


def test_single_consumer_is_one_use(p, q):
    uses = uses_of(p ^ q)
    assert one_use(uses, p)
    assert one_use(uses, q)


def test_repeated_operand_is_not_one_use(p):
    uses = uses_of(p & p)
    assert not one_use(uses, p)


def test_shared_subexpression_is_not_one_use(p, q):
    shared = p & q
    expr = shared ^ (shared | p)
    uses = uses_of(expr)
    assert not one_use(uses, shared)
    # p is consumed by `p & q` and by `shared | p`
    assert not one_use(uses, p)
    assert one_use(uses, q)


def test_uint_bits_follow_their_consumers(x, y):
    total = x + y
    uses = uses_of(total.bits[0] ^ total.bits[1])
    assert all(one_use(uses, b) for b in x.bits)
    assert all(one_use(uses, b) for b in y.bits)


def test_uint_used_twice_is_not_one_use(x, y):
    total = x + y
    uses = uses_of((total + total).bits[0])
    assert all(not one_use(uses, b) for b in total.bits)
    assert all(one_use(uses, b) for b in x.bits)


def test_one_use_does_not_affect_qubit_equality():
    raw = Qubit()
    assert Q(raw, one_use=True) == Q(raw, one_use=False)


def gadget_operand_flags(monkeypatch, expr):
    """one_use flags of each gadget's operands, in build order"""
    from qlang.compile import compile as compile_mod

    calls = []
    real_get_gadget = compile_mod.get_gadget

    def spy(node):
        gadget = real_get_gadget(node)

        def wrapped(qctx, node, *operands):
            calls.append([[q.one_use for q in qs] for qs in operands])
            return gadget(qctx, node, *operands)

        return wrapped

    monkeypatch.setattr(compile_mod, "get_gadget", spy)
    compile_mod.compile_oracle(expr)
    return calls


def test_operands_are_one_use_at_their_only_parent(monkeypatch, p, q):
    assert gadget_operand_flags(monkeypatch, p ^ q) == [[[True], [True]]]


def test_one_use_is_set_at_last_parent_not_at_startup(monkeypatch, p, q):
    # p has two parents: `p & q` (first) and `shared | p` (last); shared has two parents too
    shared = p & q
    expr = shared ^ (shared | p)
    and_flags, or_flags, xor_flags = gadget_operand_flags(monkeypatch, expr)
    assert and_flags == [[False], [True]]
    assert or_flags == [[False], [True]]
    assert xor_flags == [[True], [True]]
