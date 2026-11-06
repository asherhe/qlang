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
                qeq = next(qtemp_iter)
                if qa is not None:
                    qctx.cx(qa, qeq)
                elif qb is not None:
                    qctx.cx(qb, qeq)
                qctx.x(qeq)
                qcontrol.append(qeq)

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
    for i, qb in enumerate(b):
        qctx.x(neg_b[i])
        if i < len(b):
            qctx.cx(b[i], neg_b[i])

    carry_1 = qctx.pool.alloc()
    qctx.x(carry_1)
    qout = qctx.pool.alloc()
    diff_qubits = q_adder(qctx, node, a, neg_b, carry_in=carry_1, carry_out=qout, fill=1)
    return [qout]


def _maj(qctx: QCtx, a: Q, b: Q, c: Q) -> None:
    qctx.cx(a, b)
    qctx.cx(a, c)
    qctx.ccx(c, b, a)

def _uma(qctx: QCtx, a: Q, b: Q, c: Q) -> None:
    qctx.ccx(c, b, a)
    qctx.cx(a, c)
    qctx.cx(c, b)

def q_adder(
    qctx: QCtx,
    node: ASTNode,
    a: list[Q],
    b: list[Q],
    out: list[Q] | None = None,
    carry_in: Q | None = None,
    carry_out: Q | None = None,
    fill: int = 0,
) -> list[Q]:
    """
    construct a quantum uint addition circuit

    accepts two UInt operands, a and b. the shorter operand is treated as if it were padded with 0s

    the output is of width max(a.width, b.width). the final carry is copied into carry_out if given

    fill dictates the value the MSB of the smaller operand when it is extended
    """
    # uses cuccaro adder, 1 ancila qubit required

    if fill not in (0, 1):
        raise ValueError("fill must be 0 or 1")

    # normalize to len(a) >= len(b)
    if len(a) < len(b):
        a, b = b, a

    width = len(a)
    if width == 0:
        return []

    # determine 
    qcarry: list[Q] = []         # carry propagation qubits
    qtarget: list[Q] = []        # sum target qubits
    carry_ancillas: list[Q] = [] # ancillas allocated for garry prop
    for i in range(width):
        qa = a[i]
        qb = b[i] if i < len(b) else None
        # both operands exist
        if qb is not None:
            if qa.one_use:
                carry_i, target_i = qb, qa
            elif qb.one_use:
                carry_i, target_i = qa, qb
            else:
                # neither is one_use, copy qb into ancilla target
                carry_i = qa
                target_i = qctx.pool.alloc()
                qctx.cx(qb, target_i)
        else:
            if qa.one_use:
                carry_i = qctx.pool.alloc()
                target_i = qa
                if fill:
                    qctx.x(carry_i)
                carry_ancillas.append(carry_i)
            else:
                carry_i = qa
                target_i = qctx.pool.alloc()
                if fill:
                    qctx.x(target_i)
        qcarry.append(carry_i)
        qtarget.append(target_i)

    # carry_ancilla is always reset to its original value so we don't need to check for one_use
    if carry_in is not None and carry_in not in [*a, *b, *(out if out else ()), carry_out]:
        carry_0 = carry_in
        carry_allocated = False
    else:
        carry_0 = qctx.pool.alloc()
        carry_allocated = True
        if carry_in is not None:
            qctx.cx(carry_in, carry_0)

    # forward MAJ sweep
    for i in range(width):
        c = carry_0 if i == 0 else qcarry[i - 1]

        if i >= len(b) and fill == 0 and a[i].one_use:
            # qcarry[i] is |0>
            # MAJ(0, t, c) simplifies to CCX
            qctx.ccx(c, qtarget[i], qcarry[i])
        else:
            _maj(qctx, qcarry[i], qtarget[i], c)

    # copy carry_out state
    if carry_out is not None:
        qctx.cx(qcarry[-1], carry_out)

    # reverse UMA sweep
    for i in range(width-1, -1, -1):
        c = carry_0 if i == 0 else qcarry[i - 1]

        if i >= len(b) and fill == 0 and a[i].one_use:
            qctx.ccx(c, qtarget[i], qcarry[i])
            qctx.cx(qcarry[i], c)
            qctx.cx(c, qtarget[i])
        else:
            _uma(qctx, qcarry[i], qtarget[i], c)
    
    for qc in reversed(carry_ancillas):
        if fill:
            qctx.x(qc)
        qctx.pool.free(qc)

    if carry_allocated:
        qctx.pool.free(carry_0)

    return qtarget

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

