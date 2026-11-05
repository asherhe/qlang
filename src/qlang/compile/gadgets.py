from enum import IntEnum
from typing import Callable

from qiskit.circuit import QuantumCircuit

from ..expression import ASTNode, BitType, UIntType

from ..optimize import NodeType
from .qubit import QCtx, Q

Gadget = Callable[[QCtx, ASTNode, list[list[Q]]], list[Q]]
GADGETS: dict[tuple[type[NodeType], IntEnum], Gadget] = {}
def gadget(*kinds: NodeType):
    def deco(fn):
        for k in kinds:
            GADGETS[(type(k), k)] = fn
        return fn
    return deco

@gadget(BitType.AND)
def q_AND(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
    qubitA, qubitB = (o[0] for o in operands)
    target_qubit = qctx.pool.alloc()
    if qubitA == qubitB:
        qctx.cx(qubitA, target_qubit)
    else:
        qctx.ccx(qubitA, qubitB, target_qubit)
    return [target_qubit]


@gadget(BitType.OR)
def q_OR(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
    qubitA, qubitB = (o[0] for o in operands)
    target_qubit = qctx.pool.alloc()
    if qubitA == qubitB:
        qctx.cx(qubitA, target_qubit)
    else:
        qctx.cx(qubitA, target_qubit)
        qctx.cx(qubitB, target_qubit)
        qctx.ccx(qubitA, qubitB, target_qubit)
    return [target_qubit]

@gadget(BitType.XOR)
def q_XOR(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
    qubitA, qubitB = (o[0] for o in operands)
    target_qubit = qctx.pool.alloc()
    qctx.cx(qubitA, target_qubit)
    qctx.cx(qubitB, target_qubit)
    return [target_qubit]

@gadget(BitType.NOT)
def q_NOT(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
    qubitA, qubitB = (o[0] for o in operands)
    target_qubit = qctx.pool.alloc()
    qctx.x(target_qubit)
    qctx.cx(qubitA, target_qubit)
    return [target_qubit]

@gadget(BitType.NE)
def q_NE(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
    if len(left) == 1:
        quantumXOR(qctx, left[0], right[0], target)
        return

    differences = []
    for a, b in zip(left, right, strict=True):
        difference = fresh_scratch()
        quantumXOR(qctx, a, b, difference)
        differences.append(difference)

    any_different = differences[0]
    for i, difference in enumerate(differences[1:], start=1):
        result = target if i == len(differences) - 1 else fresh_scratch()
        quantumOR(qctx, any_different, difference, result)
        any_different = result

@gadget(BitType.LT)
def q_LT(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
    # BUG: cooked when size of left and right mismatch
    lower_less = None

    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        result = target if i == len(left) - 1 else fresh_scratch()
        not_a = fresh_scratch()
        quantumNOT(qctx, a, not_a)

        if lower_less is None:
            quantumAND(qctx, not_a, b, result)
        else:
            less_here = fresh_scratch()
            different = fresh_scratch()
            equal_here = fresh_scratch()
            carry_less = fresh_scratch()

            quantumAND(qctx, not_a, b, less_here)
            quantumXOR(qctx, a, b, different)
            quantumNOT(qctx, different, equal_here)
            quantumAND(qctx, equal_here, lower_less, carry_less)
            quantumOR(qctx, less_here, carry_less, result)

        lower_less = result

@gadget(UIntType.ADD)
def q_ADD(qctx: QCtx, node: ASTNode, operands: list[list[Q]]) -> list[Q]:
    """
    construct a quantum uint addition circuit

    PARAMETERS
    ----------
    a, b       : list of qubits, LSB-first. can be any length
    out        : list of qubits, LSB-last. length is the max of a and b
    carry_in   : qubit for carry input
    carry_out  : qubit for carry output
    """

    # uses cuccaro adder, 1 ancila cubit required
    # TODO: draper adder if in-place allowed
    
    # normalize to len(a) >= len(b)
    if len(a) < len(b):
        tmp = a
        a = b
        b = tmp
    assert(len(a) == len(out))

    # TODO: use carry_in for ancilla if it is not used anywhere else
    carry_ancilla = fresh_scratch()
    # copy carry_in state
    if carry_in is not None:
        qctx.cx(carry_in, carry_ancilla)

    # copy b onto out
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

