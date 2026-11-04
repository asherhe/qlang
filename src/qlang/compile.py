from dataclasses import dataclass, field
from enum import Enum, auto

from qiskit import QuantumCircuit

from .expression import ASTNode, Bit, BitType, UInt, UIntType

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
        for operand in operator.operands:
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
    for node in nodes.values():
        if isinstance(node, Bit) and node.bit_type == BitType.CONST and node.value:
            compute.x(qubit_allocation[id(node)])

    # translate the operation 
    for operator in operation_order:
        check_operation = operator.bit_type
        target_qubit = qubit_allocation[id(operator)]
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
    oracle_circuit = QuantumCircuit(qubit_label)
    oracle_circuit.compose(compute, inplace=True)
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
        for operand in node.operands:
            visit(operand)
        visiting.remove(node_id)
        visited.add(node_id)

    visit(expr)


def parse_input(node: Bit) -> list[Bit]:


    operation_order: list[Bit] = []
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: ASTNode) -> None:
        if not isinstance(node, Bit):
            raise TypeError("Expected decomposed Boolean Bit nodes")
        node_id = id(node)
        if node_id in visiting:
            raise ValueError("Cycle in graph Boo hoo go fix it")
        if node_id in visited: 
            return 
        if not isinstance(node.operands, tuple):
            raise TypeError("yo fix ur shit bro keep multiple arguments in a tuple")
        visiting.add(node_id)
        for operand in node.operands:
            visit(operand)
        visiting.remove(node_id)
        visited.add(node_id)
        
        is_leaf = node.bit_type in (BitType.VAR, BitType.CONST, BitType.UINT)

        if not is_leaf:
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
    

