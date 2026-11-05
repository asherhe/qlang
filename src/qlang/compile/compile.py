from __future__ import with_statement
from typing import cast

from qiskit.circuit import AncillaQubit, QuantumCircuit

from ..expression import ASTNode, Bit, BitType, UInt, UIntType, node_type
from ..optimize import optimize_expr
from .oracle import Oracle
from .gadgets import quantumNOT, quantumXOR, quantumAND, quantumOR, quantumNE, quantumLT, quantumADD
from .qubit import AncillaPool, QCtx

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
    order = parse_input(expr)
    nodes: dict[int, ASTNode] = {id(node): node for node in order}

    # Qs for gadget operands are built on demand by keying this dict
    uses = count_uses(order, expr)

    qubit_allocation: dict[int, int] = {}  # id -> qubit index
    uint_qubits: dict[int, list[int]] = {} # id -> qubits
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
                bits = (node,)
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
    pool = AncillaPool(compute)
    qctx = QCtx(compute, pool)

    def fresh_scratch() -> int:
        """callback to allocate a new ancilla qubit"""
        index = compute.num_qubits
        compute.add_bits([AncillaQubit()])
        clean_ancillas.append(index)
        return index

    # translate the operation 
    for node_id, node in nodes.items():
        kind = node_type(node)
        if node.is_leaf:
            # initialize |1> CONST bits
            if isinstance(node, Bit) and kind == BitType.CONST and node.value:
                compute.x(qubit_allocation[node_id])
        # translate operator nodes
        elif isinstance(node, Bit):
            node_qubit = qubit_allocation[node_id]
            # check if primitive operator
            if kind not in (BitType.NOT, BitType.XOR, BitType.AND, BitType.OR, BitType.NE, BitType.GT):
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
                op_qubits = [[qubit_allocation[id(b)] for b in op.bits] for op in node.operands]

                if kind == BitType.NE:
                    quantumNE(compute, op_qubits[0], op_qubits[1], node_qubit, fresh_scratch)
                else:
                    quantumLT(compute, op_qubits[0], op_qubits[1], node_qubit, fresh_scratch)

        elif isinstance(node, UInt):
            # check if primitive operator
            if kind not in (UIntType.ADD,):
                raise TypeError(f"cannot compile non-primitive bit operator {node}")

            # all uint operations are binary so far
            op_qubits = [[qubit_allocation[id(b)] for b in op.bits] for op in node.operands]
            node_qubits = [qubit_allocation[id(b)] for b in node.bits]

            if kind == UIntType.ADD:
                sum_width = max(len(q) for q in op_qubits)
                quantumADD(compute, op_qubits[0], op_qubits[1], node_qubits[:sum_width-1], fresh_scratch, carry_out=node_qubits[sum_width])
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


def count_uses(order: list[ASTNode], root: ASTNode) -> dict[int, int]:
    """
    count how many operations consume each qubit-owning Bit.

    order is the AST in topological order.

    returns a table keyed by id()
    """
    uses: dict[int, int] = {}

    def use(bit: Bit) -> None:
        uses[id(bit)] = uses.get(id(bit), 0) + 1

    for node in order:
        for operand in node.operands:
            if isinstance(operand, Bit):
                use(operand)
            elif isinstance(operand, UInt):
                for bit in operand.bits:
                    use(bit)

    # root is implicitly used once
    if isinstance(root, Bit):
        use(root)

    return uses


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
        elif isinstance(node, UInt):
            if not isinstance(node.uint_type, UIntType):
                raise TypeError("A UInt node's uint_type must be a UIntType member")
            expected = node.expected_operands
            if len(node.operands) != expected:
                raise ValueError(
                    f"{node.uint_type.name} expects {expected} operand(s), got {len(node.operands)}"
                )
        else:
            raise TypeError(f"unsupported node type {type(node).__name__}")

        # Recurse so validation also reaches deeply nested operands. A valid
        # leaf has an empty tuple, which naturally ends that branch.
        visiting.add(node_id)
        for operand in node.operands:
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


