import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterable, cast

from qiskit import QuantumCircuit
from qiskit.circuit import AncillaQubit

from .expression import ASTNode, Bit, BitType, UInt, UIntType, const_bit, node_type
from .optimize import optimize_expr

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
    if not isinstance(expr, Bit):
        raise TypeError(
            "compile_oracle expects a Bit expression; decompose UInt expressions first"
        )
    validate_expression(expr)

    expr = cast(Bit, optimize_expr(expr))

    # all ASTNodes in the expression in topological order
    # id -> ASTNode
    nodes: dict[int, ASTNode] = {id(node): node for node in parse_input(expr)}

    qubit_allocation: dict[int, int] = {}  # id -> qubit index
    uint_qubits: dict[int, list[int]] = {}       # id -> qubits
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
            # UINT bits are allocated by their parent UInt node below
            if kind != BitType.UINT:
                bits = (cast(Bit, node),)
        elif is_uint:
            if kind == UIntType.VAR:
                inputs[cast(str, node.name)] = []
            # BITS uints do not own their own bits
            if kind == UIntType.BITS:
                uint_qubits[id(node)] = [qubit_allocation[id(b)] for b in node.bits]
            else:
                bits = node.bits
                uint_qubits[id(node)] = []

        for bit in bits:
            qubit_allocation[id(bit)] = qubit_label
            match bit.bit_type:
                case BitType.CONST:
                    clean_ancillas.append(qubit_label)
                case BitType.VAR:
                    # single-bit variable
                    inputs[cast(str, bit.name)] = [qubit_label]
                case BitType.UINT:
                    parent = cast(UInt, bit.parent_uint)
                    if parent.uint_type == UIntType.VAR:
                        inputs[cast(str, parent.name)].append(qubit_label)
                    else:
                        # part of a computed uint, mark as ancilla
                        clean_ancillas.append(qubit_label)
                case _:
                    # all other (bit operators) - mark as ancilla bit
                    clean_ancillas.append(qubit_label)

            # current node is always the parent
            if is_uint:
                uint_qubits[node_id].append(qubit_label)

            qubit_label += 1

    if id(expr) in clean_ancillas:
        clean_ancillas.remove(id(expr))

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
        # a UINT bit is one qubit of its parent, so it needs no gates of its own
        if kind == BitType.UINT:
            continue
        if node.is_leaf:
            # initialize |1> CONST bits
            if isinstance(node, Bit) and kind == BitType.CONST and node.value:
                compute.x(qubit_allocation[node_id])
            # const_bit is cached, so a CONST UInt's bits are found by position
            elif isinstance(node, UInt) and kind == UIntType.CONST:
                for bit, qubit in zip(node.bits, uint_qubits[node_id], strict=True):
                    if bit.value:
                        compute.x(qubit)
        # translate operator nodes
        elif isinstance(node, Bit):
            node_qubit = qubit_allocation[node_id]
            # check if primitive operator
            if kind not in (BitType.NOT, BitType.XOR, BitType.AND, BitType.OR, BitType.NE, BitType.LT):
                raise TypeError(f"cannot compile non-primitive bit operator {node}")

            if kind == BitType.NOT:
                qubitA = qubit_allocation[id(node.operands[0])]
                quantumNOT(compute, qubitA, node_qubit)
            elif kind in (BitType.XOR, BitType.AND, BitType.OR):
                qubitA = qubit_allocation[id(node.operands[0])]
                qubitB = qubit_allocation[id(node.operands[1])]
                if kind == BitType.XOR:
                    quantumXOR(compute, qubitA, qubitB, node_qubit)
                elif kind == BitType.AND:
                    quantumAND(compute, qubitA, qubitB, node_qubit)
                elif kind == BitType.OR:
                    quantumOR(compute, qubitA, qubitB, node_qubit)
            elif kind in (BitType.NE, BitType.LT):
                # comparison operators
                left, right = comparison_qubits(node, qubit_allocation, uint_qubits, fresh_scratch)

                if kind == BitType.NE:
                    quantumNE(compute, left, right, node_qubit, fresh_scratch)
                else:
                    quantumLT(compute, left, right, node_qubit, fresh_scratch)

        elif isinstance(node, UInt):
            # check if primitive operator
            if kind not in (UIntType.ADD,):
                raise TypeError(f"cannot compile non-primitive bit operator {node}")

            # all uint operations are binary so far
            op_qubits = [uint_qubits[id(op)] for op in node.operands]
            node_qubits = uint_qubits[node_id]

            if kind == UIntType.ADD:
                sum_width = max(len(q) for q in op_qubits)
                quantumADD(compute, op_qubits[0], op_qubits[1], node_qubits[:sum_width], fresh_scratch, carry_out=node_qubits[sum_width])
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
    """Check expression structure and metadata before allocating qubits."""
    # visiting catches circular dependencies; visited skips shared nodes that
    # have already passed every check. Neither set changes the expression.
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: ASTNode) -> None:
        if not isinstance(node, (Bit, UInt)):
            raise TypeError(f"Expected a Bit or UInt node, got {type(node).__name__}")

        node_id = id(node)
        if node_id in visiting:
            raise ValueError("Cycle detected in expression graph")
        if node_id in visited:
            return

        if not isinstance(node.operands, tuple):
            raise TypeError("Node operands must be a tuple; use (operand,) for one operand")

        if isinstance(node, Bit):
            # BitType is an IntEnum: checking its class prevents an unrelated
            # integer or UIntType member with the same number from being accepted.
            if not isinstance(node.bit_type, BitType):
                raise TypeError("A Bit node's bit_type must be a BitType member")
            expected = node.expected_operands
            if len(node.operands) != expected:
                raise ValueError(
                    f"{node.bit_type.name} expects {expected} operand(s), got {len(node.operands)}"
                )
            if node.bit_type in (BitType.VAR, BitType.UINT):
                input_slot(node)
            elif node.bit_type == BitType.CONST:
                if type(node.value) is not bool:
                    raise TypeError("constant bit nodes should be true or false")
            elif node.bit_type in (BitType.NOT, BitType.AND, BitType.OR, BitType.XOR):
                if any(not isinstance(operand, Bit) for operand in node.operands):
                    raise TypeError("Boolean operations require Bit operands")
            else:
                left, right = node.operands
                if not (
                    isinstance(left, Bit) and isinstance(right, Bit)
                    or isinstance(left, UInt) and isinstance(right, UInt)
                ):
                    raise TypeError("comparison operands must both be Bit or both be UInt")
        elif isinstance(node, UInt):
            if not isinstance(node.uint_type, UIntType):
                raise TypeError("a UInt node's uint_type must be a UIntType member")
            expected = node.expected_operands
            if len(node.operands) != expected:
                raise ValueError(
                    f"{node.uint_type.name} expects {expected} operand(s), got {len(node.operands)}"
                )
            if type(node.width) is not int or node.width < 1:
                raise ValueError("UInt width must be a positive integer")
            if node.uint_type == UIntType.VAR:
                if not isinstance(node.name, str) or not node.name.strip():
                    raise ValueError("input needs a name")
            if not isinstance(node.bits, tuple):
                raise TypeError("UInt bits must be a tuple")
            if len(node.bits) != node.width:
                raise ValueError("UInt must contain exactly width bits")
            if any(not isinstance(bit, Bit) for bit in node.bits):
                raise TypeError("UInt bits must contain only Bit nodes")
            if any(not isinstance(operand, UInt) for operand in node.operands):
                raise TypeError("Integer operations require UInt operands")

            # Owned output bits point back to this UInt. Inspect their metadata
            # directly, because recursively visiting them would revisit the owner.
            # BITS wrappers instead depend on their bits and are traversed below.
            if node.uint_type == UIntType.CONST:
                for bit in node.bits:
                    if not isinstance(bit.bit_type, BitType) or bit.bit_type != BitType.CONST:
                        raise TypeError("A constant UInt must contain constant Bit nodes")
                    visit(bit)
            elif node.uint_type != UIntType.BITS:
                for index, bit in enumerate(node.bits):
                    if not isinstance(bit.bit_type, BitType) or bit.bit_type != BitType.UINT:
                        raise TypeError("Owned UInt bits must have BitType.UINT")
                    if bit.parent_uint is not node:
                        raise ValueError("Owned UInt bits must reference their parent")
                    if type(bit.index) is not int or bit.index != index:
                        raise ValueError("Owned UInt bit indices must match their positions")
                    if not isinstance(bit.operands, tuple) or bit.operands:
                        raise ValueError("Owned UInt bits must have an empty operand tuple")

        children = node.children
        if not isinstance(children, tuple):
            raise TypeError("Node children must be a tuple")

        # Follow the same dependencies as parse_input, including the bits inside
        # BITS wrappers and the parent of an indexed input bit.
        visiting.add(node_id)
        for child in children:
            visit(child)
        if isinstance(node, Bit) and node.bit_type == BitType.UINT:
            parent = cast(UInt, node.parent_uint)
            if parent.bits[cast(int, node.index)] is not node:
                raise ValueError("Indexed bit must be stored at its position in the parent")
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

def comparison_qubits(node: Bit, qubit_allocation: dict[int, int], uint_qubits: dict[int, list[int]], fresh_scratch) -> tuple[list[int], list[int]]:
    """
    temporary patch: match comparison operand widths

    the shorter operand is extended with scratch qubits, which stay |0> and so
    act as leading zero bits
    """
    left, right = (
        list(uint_qubits[id(op)]) if isinstance(op, UInt) else [qubit_allocation[id(op)]]
        for op in node.operands
    )
    width = max(len(left), len(right))
    left += [fresh_scratch() for _ in range(width - len(left))]
    right += [fresh_scratch() for _ in range(width - len(right))]
    return left, right

def quantumNE(compute: QuantumCircuit, left, right, target, fresh_scratch):
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

def quantumLT(compute: QuantumCircuit, left, right, target, fresh_scratch):
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
    
def quantumADD(compute: QuantumCircuit, a: list[int], b: list[int], out: list[int], fresh_scratch, carry_in=None, carry_out=None):
    """out = a + b. a and b are only read; out and carry_out must start at |0>"""
    if len(a) < len(b):
        a, b = b, a
    assert len(a) == len(out)

    carry = carry_in
    for i in range(len(a)):
        # the bits meeting in this column: a[i], b[i] and the carry
        column = [a[i]]
        if i < len(b):
            column.append(b[i])
        if carry is not None:
            column.append(carry)

        # sum bit = XOR of the column
        for qubit in column:
            compute.cx(qubit, out[i])

        is_last = i == len(a) - 1
        if is_last and carry_out is None:
            break

        # next carry = majority of the column: ab ^ ac ^ bc
        next_carry = carry_out if is_last else fresh_scratch()
        for first, second in itertools.combinations(column, 2):
            compute.ccx(first, second, next_carry)
        carry = next_carry
