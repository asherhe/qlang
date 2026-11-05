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
    qubitA, qubitB = a[0], b[0]
    target_qubit = qctx.pool.alloc()
    if qubitA == qubitB:
        qctx.cx(qubitA, target_qubit)
    else:
        qctx.ccx(qubitA, qubitB, target_qubit)
    return [target_qubit]


@gadget(BitType.OR)
def q_OR(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    qubitA, qubitB = a[0], b[0]
    target_qubit = qctx.pool.alloc()
    if qubitA == qubitB:
        qctx.cx(qubitA, target_qubit)
    else:
        qctx.cx(qubitA, target_qubit)
        qctx.cx(qubitB, target_qubit)
        qctx.ccx(qubitA, qubitB, target_qubit)
    return [target_qubit]

@gadget(BitType.XOR)
def q_XOR(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    qubitA, qubitB = a[0], b[0]
    target_qubit = qctx.pool.alloc()
    qctx.cx(qubitA, target_qubit)
    qctx.cx(qubitB, target_qubit)
    return [target_qubit]

@gadget(BitType.NOT)
def q_NOT(qctx: QCtx, node: ASTNode, a: list[Q]) -> list[Q]:
    qubitA = a[0]
    target_qubit = qctx.pool.alloc()
    qctx.x(target_qubit)
    qctx.cx(qubitA, target_qubit)
    return [target_qubit]

@gadget(BitType.NE)
def q_NE(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q]) -> list[Q]:
    width = max(len(a), len(b))
    differences = []
    for i in range(width):
        diff = qctx.pool.alloc()
        qA = a[i] if i < len(a) else None
        qB = b[i] if i < len(b) else None

        if qA and qB:
            qctx.cx(qA, diff)
            qctx.cx(qB, diff)
        elif qA:
            qctx.cx(qA, diff)
        elif qB:
            qctx.cx(qB, diff)
        differences.append(diff)

    # OR the differences together
    any_different = differences[0]
    for i in range(1, len(differences)):
        target = differences[i] if i == len(differences) - 1 else qctx.pool.alloc()
        qctx.cx(any_different, target)
        qctx.cx(differences[i], target)
        qctx.ccx(any_different, differences[i], target)
        any_different = target
    return [any_different]

# @gadget(BitType.LT)
# def q_LT(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
#     # BUG: cooked when size of left and right mismatch
#     lower_less = None
#
#     for i, (a, b) in enumerate(zip(left, right, strict=True)):
#         result = target if i == len(left) - 1 else fresh_scratch()
#         not_a = fresh_scratch()
#         quantumNOT(qctx, a, not_a)
#
#         if lower_less is None:
#             quantumAND(qctx, not_a, b, result)
#         else:
#             less_here = fresh_scratch()
#             different = fresh_scratch()
#             equal_here = fresh_scratch()
#             carry_less = fresh_scratch()
#
#             quantumAND(qctx, not_a, b, less_here)
#             quantumXOR(qctx, a, b, different)
#             quantumNOT(qctx, different, equal_here)
#             quantumAND(qctx, equal_here, lower_less, carry_less)
#             quantumOR(qctx, less_here, carry_less, result)
#
#         lower_less = result

def q_adder(qctx: QCtx, node: ASTNode, a: list[Q], b: list[Q], carry_in: Q | None = None, carry_out: Q | None = None) -> list[Q]:
    """
    construct a quantum uint addition circuit
    
    accepts two UInt operands, a and b. the shorter operand is treated as if it were padded with 0s

    the output is of width max(a.width, b.width) + 1, representing the sum and teh carry bit
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

    # create and initialize output ancillas
    out = qctx.pool.allocs(len(b))
    for i in range(len(b)):
        qctx.cx(b[i], out[i])

    # forward sweep
    for i in range(len(b)):
        qctx.cx(carry_ancilla, out[i])
        qctx.cx(carry_ancilla, a[i])
        qctx.ccx(a[i], out[i], carry_ancilla)
        
    # forward sweep where b is 0
    # simplifies to a toffoli gate
    for i in range(len(b), len(a)):
        qctx.cx(carry_ancilla, a[i])

    # copy carry_out state
    if carry_out is not None:
        qctx.cx(carry_ancilla, carry_out)

    # forward sweep
    for i in range(len(a)-1, len(b)-1, -1):
        qctx.cx(carry_ancilla, a[i])
        
    # forward sweep where b is 0
    # simplifies to a toffoli gate
    for i in range(len(b)-1, -1, -1):
        qctx.ccx(a[i], out[i], carry_ancilla)
        qctx.cx(carry_ancilla, a[i])
        qctx.cx(carry_ancilla, out[i])

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
        qctx.cx(neg_b[i])
        if i < len(b):
            qctx.cx(b[i], neg_b[i])

    carry_1 = qctx.pool.alloc()
    qctx.x(carry_1)
    diff_qubits = q_adder(qctx, node, a, neg_b, carry_in=carry_1)
    return diff_qubits

