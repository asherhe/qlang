from __future__ import annotations

from dataclasses import replace
from typing import cast

from qiskit.circuit import QuantumCircuit, Qubit

from ..expression import ASTNode, Bit, BitType, UInt, UIntType, node_type
from ..optimize import optimize_expr
from .gadgets import get_gadget
from .oracle import Oracle
from .qubit import AncillaPool, Q, QCtx

def compile_oracle(expr: Bit) -> Oracle:
    """Compile decomposed Boolean logic into a clean phase oracle.

    Inputs and their bit positions are recorded in inputs. Phase mode has no
    external output qubits, so output is empty. All workspace starts/ends at zero.

    this function owns the creation of every Q. only leaves (variables and constants) are
    initialized up front; every other node is built on the fly by its gadget, and the Qs the
    gadget returns are annotated with one_use here.
    """
    if not isinstance(expr, Bit):
        raise TypeError(
            "compile_oracle expects a Bit expression; decompose UInt expressions first"
        )
    validate_expression(expr)

    expr = cast(Bit, optimize_expr(expr))

    # all ASTNodes in the expression in topological order
    order = parse_input(expr)
    # how many operations consume each Bit's qubit, keyed by id(Bit)
    uses = count_uses(order, expr)

    circuit = QuantumCircuit()
    qctx = QCtx(circuit)

    # id(node) -> qubits of the node. a Bit has one, a UInt has one per bit (LSB first)
    qubits: dict[int, list[Q]] = {}
    # var name -> qubits of the node
    inputs: dict[str, list[Q]] = {}

    def tag(q: Q, bit: Bit) -> Q:
        """update Q one_use metadata for a bit"""
        return replace(q, one_use=not q.is_const and uses.get(id(bit), 0) == 1)

    def new_variable(bit: Bit) -> Q:
        raw = Qubit()
        circuit.add_bits([raw])
        return tag(Q(raw), bit)

    def init_const(bit: Bit) -> Q:
        # CONST bits are shared by the node cache, so only ever initialize one once
        # the Qubit represented by this is a reusable qubit that 
        if id(bit) not in qubits:
            qubits[id(bit)] = [qctx.const_one if bit.value else qctx.const_zero]
        return qubits[id(bit)][0]

    # initialize leaves - variables and constants are the only nodes not built by a gadget
    for node in order:
        if not node.is_leaf:
            continue
        if isinstance(node, Bit):
            if node.bit_type == BitType.CONST:
                init_const(node)
            elif node.bit_type == BitType.VAR:
                qubits[id(node)] = [new_variable(node)]
                inputs[cast(str, node.name)] = qubits[id(node)]
        elif isinstance(node, UInt):
            if node.uint_type == UIntType.CONST:
                qubits[id(node)] = [init_const(b) for b in node.bits]
            elif node.uint_type == UIntType.VAR:
                bit_qs = [new_variable(b) for b in node.bits]
                for b, q in zip(node.bits, bit_qs):
                    qubits[id(b)] = [q]
                qubits[id(node)] = bit_qs
                inputs[cast(str, node.name)] = bit_qs

    # translate the operations by running each node's gadget
    for node in order:
        if node.is_leaf:
            continue
        if isinstance(node, Bit) and node.bit_type == BitType.UINT:
            # set when the parent UInt was initialized or built
            continue
        if isinstance(node, UInt) and node.uint_type == UIntType.BITS:
            # BITS uints alias the qubits of their component bits
            qubits[id(node)] = [qubits[id(b)][0] for b in node.bits]
            continue

        gadget = get_gadget(node)
        out = gadget(qctx, node, *(qubits[id(op)] for op in node.operands))

        out_bits: tuple[Bit, ...]
        if isinstance(node, UInt):
            expected_width = node.width
            out_bits = node.bits
        else:
            expected_width = 1
            out_bits = (cast(Bit, node),)
        if len(out) != expected_width:
            raise RuntimeError(
                f"gadget for {node_type(node).name} returned {len(out)} qubit(s), expected {expected_width}"
            )

        tagged = [tag(q, b) for q, b in zip(out, out_bits)]
        qubits[id(node)] = tagged
        if isinstance(node, UInt):
            for b, q in zip(out_bits, tagged):
                qubits[id(b)] = [q]

    # Calculate the predicate, mark its phase, then undo all scratch work.
    predicate = qubits[id(expr)][0]
    oracle_circuit = circuit.copy()
    oracle_circuit.z(predicate.qubit)
    oracle_circuit.compose(circuit.inverse(), inplace=True)

    def index(q: Q) -> int:
        return circuit.find_bit(q.qubit).index

    return Oracle(
        circuit=oracle_circuit,
        inputs={name: [index(q) for q in bit_qs] for name, bit_qs in inputs.items()},
        output=[],
        clean_ancillas=[index(q) for q in qctx.pool.ancillas],
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


