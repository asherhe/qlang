from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, cast

from qiskit import QuantumCircuit
from qiskit.circuit import AncillaQubit

from .expression import ASTNode, Bit, BitType, UInt, UIntType, const_bit, node_type

class OutputMode(Enum):
    BIT_FLIP = auto()
    PHASE = auto()

@dataclass
class Oracle:
    """container class for compiled oracle circuit outputs"""

    # compiled circuit result
    circuit: QuantumCircuit
    
    # variable name -> qubit indices
    # orphan Bits (not in a UInt) are treated as 1-bit width UInts
    inputs: dict[str, list[int]] 

    # External outputs; empty for compile_oracle's phase-only interface.
    output: list[tuple[int, OutputMode]]

    # track ancilla qubits
    clean_ancillas: list[int] = field(default_factory=list)
    dirty_ancillas: list[int] = field(default_factory=list)

    @property
    def total_ancillas(self) -> int:
        return len(self.clean_ancillas) + len(self.dirty_ancillas)

def compile_oracle(expr: Bit) -> Oracle:
    """Compile decomposed Boolean logic into a clean phase oracle.

    Inputs and their bit positions are recorded in inputs. Phase mode has no
    external output qubits, so output is empty. All workspace starts/ends at zero.
    """
    validate_expression(expr)

    # all ASTNodes in the expression in topological order
    # id -> ASTNode
    nodes: dict[int, ASTNode] = {id(node): node for node in parse_input(expr)}

    qubit_allocation: dict[int, int] = {}  # id -> qubit index
    inputs: dict[str, list[int]] = {}      # input name -> qubit indices
    clean_ancillas: list[int] = []         # list[qubit id]
    qubit_label = 0

    # construct registry of qubits
    # each ASTNode contributes only itself to the registry, as its child nodes
    # will have already been iterated through
    for node_id, node in nodes.items():
        kind = node_type(node)
        is_bit = isinstance(node, Bit)
        is_uint = isinstance(node, UInt)

        bits: tuple[Bit, ...] = ()
        if is_bit:
            bits = (node,)
        elif is_uint:
            if kind == UIntType.VAR:
                inputs[cast(str, node.name)] = []
            # BITS uints do not own their own bits
            if kind != UIntType.BITS:
                bits = node.bits

        for bit in bits:
            qubit_allocation[id(bit)] = qubit_label
            match bit.bit_type:
                case BitType.CONST:
                    clean_ancillas.append(qubit_label)
                case BitType.VAR:
                    # single-bit variable
                    inputs[cast(str, bit.name)] = [qubit_label]
                case BitType.UINT:
                    assert(is_uint)
                    if node.uint_type == UIntType.VAR:
                        inputs[cast(str, node.name)].append(qubit_label)
                    elif len(node.operands) > 0:
                        # part of an operator uint, mark as ancilla
                        clean_ancillas.append(qubit_label)
                case _:
                    # all other (bit operators) - mark as ancilla bit
                    clean_ancillas.append(qubit_label)
            qubit_label += 1

    compute = QuantumCircuit(qubit_label)

    def fresh_scratch() -> int:
        """callback to allocate a new ancilla qubit"""
        index = compute.num_qubits
        compute.add_bits([AncillaQubit()])
        clean_ancillas.append(index)
        return index

    # translate the operation 
    for node_id, node in nodes.items():
        kind = node_type(node)
        node_qubit = qubit_allocation[node_id]
        if node.is_leaf:
            # initialize |1> CONST bits
            if isinstance(node, Bit) and kind == BitType.CONST and node.value:
                compute.x(node_qubit)
        # translate operator nodes
        elif isinstance(node, Bit):
            # check if primitive operator
            if kind not in (BitType.NOT, BitType.XOR, BitType.AND, BitType.NE, BitType.GT):
                raise TypeError(f"cannot compile non-primitive bit operator {node}")

            if kind in (BitType.NE, BitType.LT):
                left, right = comparison_bits(node)
                left_qubits = [qubit_allocation[id(bit)] for bit in left]
                right_qubits = [qubit_allocation[id(bit)] for bit in right]

                if kind == BitType.NE:
                    quantumNE(compute, left_qubits, right_qubits, node_qubit, fresh_scratch)
                else:
                    quantumLT(compute, left_qubits, right_qubits, node_qubit, fresh_scratch)
            else:
                qubitA = qubit_allocation[id(node.operands[0])]
                if kind == BitType.NOT:
                    quantumNOT(compute, qubitA, node_qubit)
                else:
                    qubitB = qubit_allocation[id(node.operands[1])]
                    if kind == BitType.AND:
                        quantumAND(compute, qubitA, qubitB, node_qubit)
                    elif kind == BitType.OR:
                        quantumOR(compute, qubitA, qubitB, node_qubit)
                    elif kind == BitType.XOR:
                        quantumXOR(compute, qubitA, qubitB, node_qubit)
                    else:
                        raise ValueError(f"Unsupported operation: {kind}")
        elif isinstance(node, UInt):
            raise NotImplementedError
        else:
            raise TypeError(f"compilation of {type(node)} nodes not supported")

    # Calculate the predicate, mark its phase, then undo all scratch work.
    predicate_qubit = qubit_allocation[id(expr)]
    oracle_circuit = compute.copy()
    oracle_circuit.z(predicate_qubit)
    oracle_circuit.compose(compute.inverse(), inplace=True)

    return Oracle(
        circuit=oracle_circuit,
        inputs=inputs,
        output=[],  
        clean_ancillas=clean_ancillas,
        dirty_ancillas=[],
    )


def validate_expression(expr: Bit) -> None:
    """Check that every reachable node is supported before allocating qubits."""
    # visiting catches circular dependencies; visited skips shared nodes that
    # have already passed every check. Neither set changes the expression.
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: ASTNode) -> None:
        node_id = id(node)
        if node_id in visiting:
            raise ValueError("Cycle detected in expression graph")
        if node_id in visited:
            return

        children = node.children
        if not isinstance(children, tuple):
            raise TypeError("Node children must be a tuple; use (operand,) for one operand")

        expected = node.expected_operands
        if len(children) != expected:
            raise ValueError(
                f"{node} expects {expected} operand(s), got {len(node.operands)}"
            )

        if isinstance(node, Bit):
            if node.bit_type in (BitType.VAR, BitType.UINT):
                input_slot(node)
            elif node.bit_type == BitType.CONST:
                if type(node.value) is not bool:
                    raise TypeError("constant bit nodes should be true or false")

        # Recurse so validation also reaches deeply nested operands. A valid
        # leaf has an empty tuple, which naturally ends that branch.
        visiting.add(node_id)
        for operand in node.children:
            visit(operand)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(expr)


def parse_input(node: ASTNode) -> list[ASTNode]:
    """
    topological sort of an ASTNode graph. contains all nodes in the graph,
    ensuring that a parent node only appears after all of its children.
    """

    operation_order: list[ASTNode] = []
    visiting: set[int] = set()
    visited: set[int] = set()
    def visit(node: ASTNode) -> None:
        node_id = id(node)
        if node_id in visiting:
            raise ValueError("Cycle in graph Boo hoo go fix it")
        if node_id in visited: 
            return 

        children = node.children
        if not isinstance(children, tuple):
            raise TypeError("yo fix ur shit bro keep multiple arguments in a tuple")
        visiting.add(node_id)
        for child in children:
            visit(child)
        visiting.remove(node_id)

        visited.add(node_id)
        operation_order.append(node)

    visit(node)
    return operation_order

def input_slot(node: Bit) -> tuple[str, int, int]:
    if node.bit_type == BitType.VAR:
        name, width, index = node.name, 1, 0
    elif node.bit_type == BitType.UINT:
        parent = node.parent_uint
        if not isinstance(parent, UInt) or parent.uint_type != UIntType.VAR:
            raise ValueError("lower computed integer bits before compiling")
        name, width, index = parent.name, parent.width, node.index
    else:
        raise ValueError("expected an input bit")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("input needs a name")
    if type(width) is not int or width < 1:
        raise ValueError("input width must be a positive integer")
    if type(index) is not int or not 0 <= index < width:
        raise ValueError("input bit index is out of range")

    return name, width, index


def comparison_bits(node: Bit):
    left,right = node.operands
    if isinstance(left, Bit) and isinstance(right, Bit):
        return (left,), (right,)
    if not isinstance(left, UInt) or not isinstance(right, UInt):
        raise TypeError("Comparison operands must both be Bit or both be UInt")
    width = max(left.width, right.width)
    zero = const_bit(False)
    return (left.bits + (zero,) * (width - left.width), right.bits + (zero,) * (width - right.width),)


def circuit_operands(node: Bit):
    if node.bit_type in (BitType.NE, BitType.LT):
        left, right = comparison_bits(node)
        return left + right

    return node.operands


def quantumAND(compute: QuantumCircuit,qubitA: int, qubitB: int, target_qubit: int) -> None:
    if qubitA == qubitB:
        compute.cx(qubitA, target_qubit)
    else:
        compute.ccx(qubitA, qubitB, target_qubit)


def quantumOR(compute: QuantumCircuit, qubitA: int, qubitB: int, target_qubit: int) -> None:
    if qubitA == qubitB:
        compute.cx(qubitA, target_qubit)
    else:
        compute.cx(qubitA, target_qubit)
        compute.cx(qubitB, target_qubit)
        compute.ccx(qubitA, qubitB, target_qubit)

def quantumXOR(compute: QuantumCircuit, qubitA: int, qubitB: int, target_qubit: int) -> None:
    compute.cx(qubitA, target_qubit)
    compute.cx(qubitB, target_qubit)

def quantumNOT(compute: QuantumCircuit, qubitA: int, target_qubit: int) -> None:
    compute.x(target_qubit)
    compute.cx(qubitA, target_qubit)

def quantumNE(compute, left, right, target, fresh_scratch):
    if len(left) == 1:
        quantumXOR(compute, left[0], right[0], target)
        return

    differences = []
    for a, b in zip(left, right, strict=True):
        difference = fresh_scratch()
        quantumXOR(compute, a, b, difference)
        differences.append(difference)

    any_different = differences[0]
    for i, difference in enumerate(differences[1:], start=1):
        result = target if i == len(differences) - 1 else fresh_scratch()
        quantumOR(compute, any_different, difference, result)
        any_different = result

def quantumLT(compute, left, right, target, fresh_scratch):
    lower_less = None

    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        result = target if i == len(left) - 1 else fresh_scratch()
        not_a = fresh_scratch()
        quantumNOT(compute, a, not_a)

        if lower_less is None:
            quantumAND(compute, not_a, b, result)
        else:
            less_here = fresh_scratch()
            different = fresh_scratch()
            equal_here = fresh_scratch()
            carry_less = fresh_scratch()

            quantumAND(compute, not_a, b, less_here)
            quantumXOR(compute, a, b, different)
            quantumNOT(compute, different, equal_here)
            quantumAND(compute, equal_here, lower_less, carry_less)
            quantumOR(compute, less_here, carry_less, result)

        lower_less = result
    

