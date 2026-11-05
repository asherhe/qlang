from enum import IntEnum
from typing import Callable

from qiskit.circuit import QuantumCircuit

from ..expression import ASTNode, BitType, UIntType, node_type

from ..optimize import NodeType
from .qubit import QCtx, Q

Gadget = Callable[[QCtx, ASTNode, *tuple[list[Q], ...]], list[Q]]
_GADGETS: dict[tuple[type[NodeType], IntEnum], Gadget] = {}
def gadget(*kinds: NodeType):
    def deco(fn):
        for k in kinds:
            _GADGETS[(type(k), k)] = fn
        return fn
    return deco

def get_gadget(node: ASTNode) -> Gadget:
    kind = node_type(node)
    gadget = _GADGETS.get((type(kind), kind))
    if gadget is None:
        raise TypeError(f"no gadget available to compile {node}")
    return gadget


@gadget(BitType.AND)
def q_AND(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    qa, qb = a[0], b[0]
    # in-place is not reversible so we must tank the ancilla
    qout = qctx.pool.alloc()
    qctx.ccx(qa, qb, qout)
    return [qout]


@gadget(BitType.OR)
def q_OR(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    qa, qb = a[0], b[0]
    # in-place is not reversible so we must tank the ancilla
    qout = qctx.pool.alloc()
    qctx.x(qa)
    qctx.x(qb)
    qctx.ccx(qa, qb, qout)
    # only recycle qa, qb if necessary
    if not qa.one_use:
        qctx.x(qa)
    if not qb.one_use:
        qctx.x(qb)
    return [qout]

@gadget(BitType.XOR)
def q_XOR(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    qa, qb = a[0], b[0]
    qout = qctx.pick_one_use(qa, qb)
    if qout != qa:
        qctx.cx(qa, qout)
    if qout != qb:
        qctx.cx(qb, qout)
    return [qout]

@gadget(BitType.NOT)
def q_NOT(qctx: QCtx, node: ASTNode, a: list[Q]) -> list[Q]:
    qa = a[0]
    qout = qctx.pick_one_use(qa)
    qctx.x(qout)
    if qout != qa:
        qctx.cx(qa, qout)
    return [qout]

@gadget(BitType.EQ)
def q_EQ(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    width = max(len(a), len(b))
    bitwise_eq = []
    for i in range(width):
        qEq = qctx.pool.alloc()
        qA = a[i] if i < len(a) else None
        qB = b[i] if i < len(b) else None

        if qA and qB:
            qctx.cx(qA, qEq)
            qctx.cx(qB, qEq)
        elif qA:
            qctx.cx(qA, qEq)
        elif qB:
            qctx.cx(qB, qEq)
        qctx.x(qEq)
        bitwise_eq.append(qEq)

    out_qubit = qctx.pool.alloc()
    qctx.mcx(bitwise_eq, out_qubit)
    return [out_qubit]

@gadget(BitType.LT)
def q_LT(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    diff_qubits = q_SUB(qctx, node, a, b)
    return diff_qubits[-1:]

def q_adder(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q], carry_in: Q | None = None, carry_out: Q | None = None) -> list[Q]:
    """
    construct a quantum uint addition circuit

    accepts two UInt operands, a and b. the shorter operand is treated as if it were padded with 0s

    the output is of width max(a.width, b.width). the final carry is copied into carry_out if given
    """
    # uses cuccaro adder, 1 ancila cubit required
    # TODO: draper adder if in-place allowed

    # normalize to len(a) >= len(b)
    if len(a) < len(b):
        tmp = a
        a = b
        b = tmp

    # TODO: use carry_in for ancilla if it is not used anywhere else
    carry_ancilla = qctx.pool.alloc()
    # copy carry_in state
    if carry_in is not None:
        qctx.cx(carry_in, carry_ancilla)

    # output ancillas hold b, zero-padded up to the width of a: the ancillas above len(b) start at 0
    out = qctx.pool.allocs(len(a))
    for i in range(len(b)):
        qctx.cx(b[i], out[i])

    # forward sweep (MAJ): out[i] = b[i]^c[i], a[i] = a[i]^c[i], and the carry becomes c[i+1]
    for i in range(len(a)):
        qctx.cx(carry_ancilla, out[i])
        qctx.cx(carry_ancilla, a[i])
        qctx.ccx(a[i], out[i], carry_ancilla)

    # copy carry_out state
    if carry_out is not None:
        qctx.cx(carry_ancilla, carry_out)

    # backward sweep (UMA): restore the carry to c[i] and a[i], and turn out[i] into the sum bit
    for i in range(len(a)-1, -1, -1):
        qctx.ccx(a[i], out[i], carry_ancilla)
        qctx.cx(carry_ancilla, a[i])
        qctx.cx(a[i], out[i])

    return out

@gadget(UIntType.ADD)
def q_ADD(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    carry_qubit = qctx.pool.alloc()
    sum_qubits = q_adder(qctx, node, a, b, carry_out=carry_qubit)
    sum_qubits.append(carry_qubit)
    return sum_qubits

@gadget(UIntType.SUB)
def q_SUB(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    # TODO: this is mega unoptimized
    width = max(len(a), len(b))
    neg_b = qctx.pool.allocs(width)
    # pad neg_b with ones if necessary
    for i in range(width):
        qctx.x(neg_b[i])
        if i < len(b):
            qctx.cx(b[i], neg_b[i])

    carry_1 = qctx.pool.alloc()
    qctx.x(carry_1)
    diff_qubits = q_adder(qctx, node, a, neg_b, carry_in=carry_1)
    return diff_qubits

