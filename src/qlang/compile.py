from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.circuit import AncillaQubit

from .expression import ASTNode, Bit, BitType, UInt, UIntType, const_bit

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
    operation_order = parse_input(expr)

    # Gather every node once. Operations keep their operands; include the root
    # as well so variable-only and constant-only expressions work.
    nodes: dict[int, ASTNode] = {id(expr): expr}
    for operator in operation_order:
        for operand in circuit_operands(operator):
            nodes[id(operand)] = operand

    qubit_allocation: dict[int, int] = {}
    inputs: dict[str, list[int]] = {}
    clean_ancillas: list[int] = []
    qubit_label = 0

    #grouping integers 
    for node in nodes.values():
        if not isinstance(node, Bit):
            continue
        if node.bit_type not in (BitType.VAR, BitType.UINT):
            continue

        name, width, index = input_slot(node)
        register = inputs.get(name)

        if register is None:
            register = list(range(qubit_label, qubit_label + width))
            inputs[name] = register
            qubit_label += width
        elif len(register) != width:
            raise ValueError(f"conflicting widths for {name!r}")

        qubit_allocation[id(node)] = register[index]
    for node in nodes.values():
        if isinstance(node, Bit) and node.bit_type == BitType.CONST:
            qubit_allocation[id(node)] = qubit_label
            clean_ancillas.append(qubit_label)
            qubit_label += 1
            
    for operator in operation_order:
        qubit_allocation[id(operator)] = qubit_label
        clean_ancillas.append(qubit_label)
        qubit_label += 1

    compute = QuantumCircuit(qubit_label)

    def fresh_scratch() -> int:
        index = compute.num_qubits
        compute.add_bits([AncillaQubit()])
        clean_ancillas.append(index)
        return index

    for node in nodes.values():
        if isinstance(node, Bit) and node.bit_type == BitType.CONST and node.value:
            compute.x(qubit_allocation[id(node)])

    # translate the operation 
    for operator in operation_order:
        check_operation = operator.bit_type
        target_qubit = qubit_allocation[id(operator)]

        if check_operation in (BitType.NE, BitType.LT):
            left, right = comparison_bits(operator)
            left_qubits = [qubit_allocation[id(bit)] for bit in left]
            right_qubits = [qubit_allocation[id(bit)] for bit in right]

            if check_operation == BitType.NE:
                quantumNE(compute, left_qubits, right_qubits, target_qubit, fresh_scratch)
            else:
                quantumLT(compute, left_qubits, right_qubits, target_qubit, fresh_scratch)

            continue
        qubitA = qubit_allocation[id(operator.operands[0])]
        if check_operation == BitType.NOT:
            quantumNOT(compute, qubitA, target_qubit)
        else:
            qubitB = qubit_allocation[id(operator.operands[1])]
            if check_operation == BitType.AND:
                quantumAND(compute, qubitA, qubitB, target_qubit)
            elif check_operation == BitType.OR:
                quantumOR(compute, qubitA, qubitB, target_qubit)
            elif check_operation == BitType.XOR:
                quantumXOR(compute, qubitA, qubitB, target_qubit)
            else:
                raise ValueError(f"Unsupported operation: {check_operation}")

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
    # Dictionary: supported operation -> required number of child nodes.
    # Leaves have no children; NOT has one; the other operations have two.
    operand_counts = {
        BitType.VAR: 0,
        BitType.CONST: 0,
        BitType.UINT: 0,
        BitType.NOT: 1,
        BitType.AND: 2,
        BitType.OR: 2,
        BitType.XOR: 2,
        BitType.NE: 2,
        BitType.LT:2

    }

    # Track integer IDs instead of objects because nodes overload equality.
    # visiting catches circular dependencies; visited skips shared nodes that
    # have already passed every check. Neither set changes the expression.
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: ASTNode) -> None:
        # Check the class before reading .bit_type. UInt has .uint_type instead
        # and must be decomposed by stage 2 before reaching this compiler.
        if not isinstance(node, Bit):
            raise TypeError("Expected a Bit node; decompose UInt expressions in stage 2 first")

        node_id = id(node)
        if node_id in visiting:
            raise ValueError("Cycle detected in expression graph")
        if node_id in visited:
            return

        # BitType is an IntEnum: checking its class prevents an unrelated
        # integer or UIntType member with the same number from being accepted.
        if not isinstance(node.bit_type, BitType):
            raise TypeError("A Bit node's bit_type must be a BitType member")
        if node.bit_type not in operand_counts:
            raise ValueError(f"Unsupported operation {node.bit_type.name}; decompose it in stage 2 first")
        if not isinstance(node.operands, tuple):
            raise TypeError("Node operands must be a tuple; use (operand,) for one operand")

        expected = operand_counts[node.bit_type]
        if len(node.operands) != expected:
            raise ValueError(
                f"{node.bit_type.name} expects {expected} operand(s), got {len(node.operands)}"
            )

        if node.bit_type in (BitType.VAR, BitType.UINT):
            input_slot(node)
        elif node.bit_type == BitType.CONST:
            if type(node.value) is not bool:
                raise TypeError("constant bit nodes should be true or false")

        # Recurse so validation also reaches deeply nested operands. A valid
        # leaf has an empty tuple, which naturally ends that branch.
        visiting.add(node_id)
        for operand in circuit_operands(node):
            visit(operand)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(expr)


def parse_input(node: ASTNode) -> list[ASTNode]:
    """topological sort of an ASTNode graph"""

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
        
        if len(children) > 0:
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
    

