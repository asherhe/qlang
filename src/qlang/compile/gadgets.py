from enum import IntEnum
from typing import Callable, Literal, cast

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
    if qa == qb:
        if qa.one_use:
            return [qa]
        else:
            qout = qctx.pool.alloc()
            qctx.cx(qa, qout)
            return [qout]

    # in-place is not reversible so we must tank the ancilla
    qout = qctx.pool.alloc()
    qctx.cx(qa, qout)
    qctx.cx(qb, qout)
    qctx.ccx(qa, qb, qout)
    return [qout]

@gadget(BitType.XOR)
def q_XOR(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    qa, qb = a[0], b[0]
    if qa == qb:
        return [qctx.pool.alloc()]

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
    qeq = []

    # list[ qubit use policy, qa, qb, qout ]
    spec: list[tuple[Literal["true", "reuse", "ancilla"], Q | None, Q | None, Q | None]] = []
    num_ancilla: int = 0
    # index bits to see which are usable
    for i in range(width):
        qa = a[i] if i < len(a) else None
        qb = b[i] if i < len(b) else None

        if qa is not None and qb and qa == qb:
            spec.append(("true", qa, qb, None))
        elif qa is not None and qa.one_use:
            spec.append(("reuse", qa, qb, qa))
        elif qb is not None and qb.one_use:
            spec.append(("reuse", qb, qa, qb))
        else:
            spec.append(("ancilla", qa, qb, None))
            num_ancilla += 1

    qcontrol: list[Q] = []
    with qctx.pool.scratch(num_ancilla, 1) as (qtemps, qeq):
        qtemp_iter = iter(qtemps)
        qeq = qeq[0]
        for kind, qa, qb, qout in spec:
            qout = cast(Q, qout)
            if kind == "true":
                pass
            elif kind == "reuse":
                qother = qb if qout == qa else qa
                if qother is not None:
                    qctx.cx(qother, qout)
                qctx.x(qout)
                qcontrol.append(qout)
            elif kind == "ancilla":
                qtemp = next(qtemp_iter)
                if qa is not None:
                    qctx.cx(qa, qtemp)
                if qb is not None:
                    qctx.cx(qb, qtemp)
                qctx.x(qtemp)
                qcontrol.append(qtemp)

        if qcontrol:
            qctx.mcx(qcontrol, qeq)
        else:
            qctx.x(qeq) # just write |1>

    return [qeq]

@gadget(BitType.GE)
def q_GE(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    width = max(len(a), len(b))
    neg_b = qctx.pool.allocs(width)
    # pad neg_b with ones if necessary
    for i in range(width):
        qctx.x(neg_b[i])
        if i < len(b):
            qctx.cx(b[i], neg_b[i])

    carry_1 = qctx.pool.alloc()
    qctx.x(carry_1)
    qout = qctx.pool.alloc()
    diff_qubits = q_adder(qctx, node, a, neg_b, carry_in=carry_1, carry_out=qout)
    return [qout]

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

    # the sweeps below change a in place, so repeated qubits (shared const bits) need copies
    seen: set[Q] = set()
    a = list(a)
    for i, qa in enumerate(a):
        if qa in seen:
            a[i] = qctx.pool.alloc()
            qctx.cx(qa, a[i])
        seen.add(qa)

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

