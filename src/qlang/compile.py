from dataclasses import dataclass, field
from enum import Enum, auto

from qiskit import QuantumCircuit

from .expression import ASTNode, Bit, BitType

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

    # Group input nodes by name before allocating. x[0] and x[1] belong to the
    # same integer register, rather than being conflicting standalone inputs.
    variable_groups: dict[str, list[Bit]] = {}
    for node in nodes.values():
        if isinstance(node, Bit) and node.bit_type == BitType.VAR:
            name = node.name
            assert name is not None  # validate_expression already checked this.
            variable_groups.setdefault(name, []).append(node)

    for name, variables in variable_groups.items():
        indexed = variables[0].index is not None
        if any((node.index is not None) != indexed for node in variables):
            raise ValueError(f"Input {name!r} mixes standalone and indexed bits")

        width = 1
        if indexed:
            # Metadata must preserve the original bit index. Even if only x[2]
            # is referenced, reserve lower bits and the full declared width.
            width = max(node.index for node in variables if node.index is not None) + 1
            declared_widths = {node.parent_width for node in variables if node.parent_width is not None}
            if len(declared_widths) > 1:
                raise ValueError(f"Input {name!r} has conflicting parent widths")
            if declared_widths:
                declared_width = next(iter(declared_widths))
                if width > declared_width:
                    raise ValueError(f"Input {name!r} has a bit outside its declared width")
                width = declared_width

        inputs[name] = list(range(qubit_label, qubit_label + width))
        qubit_label += width
        for node in variables:
            position = 0 if node.index is None else node.index
            # Separate objects naming the same input bit share its qubit.
            qubit_allocation[id(node)] = inputs[name][position]

    # Constants and operation results are workspace, not search inputs.
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
            # Each constant object appears once in nodes, so X is applied once.
            compute.x(qubit_allocation[id(node)])

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
        output=[],  # Phase-only interface: no persistent Boolean output.
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

        if node.bit_type == BitType.VAR:
            if not isinstance(node.name, str) or not node.name.strip():
                raise ValueError("Variable nodes need a nonempty name")
            # An indexed input represents one bit of a UInt. A Boolean is not
            # a valid index even though Python treats bool as a subclass of int.
            if node.index is not None and (type(node.index) is not int or node.index < 0):
                raise ValueError("Variable bit indices must be nonnegative integers")
            if node.parent_width is not None:
                if type(node.parent_width) is not int or node.parent_width <= 0:
                    raise ValueError("Variable parent_width must be a positive integer")
                if node.index is None or node.index >= node.parent_width:
                    raise ValueError("Variable bit index must be within parent_width")
        elif node.bit_type == BitType.CONST:
            # Read the constant's value, not the symbolic node's truth value.
            if type(node.value) is not bool:
                raise TypeError("Constant Bit nodes must contain True or False")

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
        # recursively visit the the next nodes to get the operations on inputs
        # hopefully builds them in order

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
        
        is_leaf = node.bit_type in (BitType.VAR, BitType.CONST)

        if not is_leaf:
            operation_order.append(node)

    visit(node)
    return operation_order

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
    

