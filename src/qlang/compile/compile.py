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
    initialized up front; every other node is built on the fly by its gadget. a Q is marked
    one_use only when the node being built is the last parent of its bit that has not been built yet.
    """
    if not isinstance(expr, Bit):
        raise TypeError(
            "compile_oracle expects a Bit expression; decompose UInt expressions first"
        )
    validate_expression(expr)

    expr = cast(Bit, optimize_expr(expr))

    # all ASTNodes in the expression in topological order
    order = parse_input(expr)
    # how many operations still have to consume each Bit's qubit, keyed by id(Bit).
    # starts at the total use count and goes down as each parent node is built
    remaining = count_uses(order, expr)

    circuit = QuantumCircuit()
    qctx = QCtx(circuit)

    # id(node) -> qubits of the node. a Bit has one, a UInt has one per bit (LSB first)
    qubits: dict[int, list[Q]] = {}
    # var name -> qubits of the node
    inputs: dict[str, list[Q]] = {}
    # symbolic gadget -> its contiguous forward gate interval. This is kept as
    # optional visualization metadata; it is not used during oracle execution.
    operation_spans: list[tuple[str, int, int]] = []

    def expr_bits(expr: ASTNode) -> tuple[Bit, ...]:
        return expr.bits if isinstance(expr, UInt) else (cast(Bit, expr),)

    def flagged(op: ASTNode) -> list[Q]:
        """qubits of an operand, marked one_use when the parent being built is its last parent"""
        return [
            replace(q, one_use=not q.is_const and remaining.get(id(b), 0) == 1)
            for q, b in zip(qubits[id(op)], expr_bits(op))
        ]

    def release(op: ASTNode) -> None:
        """record that one parent of the operand's bits has been built"""
        for b in expr_bits(op):
            remaining[id(b)] -= 1

    def new_variable(bit: Bit) -> Q:
        raw = Qubit()
        circuit.add_bits([raw])
        return Q(raw)

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
            for op in node.operands:
                release(op)
            continue
        if isinstance(node, UInt) and node.uint_type == UIntType.BITS:
            # BITS uints alias the qubits of their component bits
            for op in node.operands:
                release(op)
            qubits[id(node)] = [qubits[id(b)][0] for b in node.bits]
            continue

        # flag every operand before releasing any, so a node that uses one operand twice
        # does not mark it one_use on the first of those uses
        operand_qs = [flagged(op) for op in node.operands]
        for op in node.operands:
            release(op)

        gadget = get_gadget(node)
        gate_start = len(circuit.data)
        out = gadget(qctx, node, *operand_qs)
        gate_end = len(circuit.data)
        if gate_end > gate_start:
            operation_spans.append((node_type(node).name, gate_start, gate_end))

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

        # one_use is set when a parent consumes this node, so the flag here is not meaningful yet
        built = [replace(q, one_use=False) for q in out]
        qubits[id(node)] = built
        if isinstance(node, UInt):
            for b, q in zip(out_bits, built):
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
        operation_spans=operation_spans,
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

        if not isinstance(node, (Bit, UInt)):
            raise TypeError(f"unsupported node type {type(node).__name__}")
        if not isinstance(node.operands, tuple):
            raise TypeError("node operands must be a tuple; use (operand,) for one operand")

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
                # bits of a computed UInt are valid, but they aren't inputs
                parent = node.parent_uint
                if not (isinstance(parent, UInt) and parent.uint_type != UIntType.VAR):
                    input_slot(node)
                if node.bit_type == BitType.UINT:
                    parent = cast(UInt, node.parent_uint)
                    if not isinstance(parent.bits, tuple):
                        raise TypeError("UInt bits must be a tuple")
                    if len(parent.bits) != parent.width:
                        raise ValueError("UInt bits must match its width")
                    if parent.bits[cast(int, node.index)] is not node:
                        raise ValueError("indexed bit must be stored at its position in its parent")
            elif node.bit_type == BitType.CONST:
                if type(node.value) is not bool:
                    raise TypeError("constant bit nodes should be true or false")
            if node.operands:
                if node.bit_type in (BitType.NOT, BitType.AND, BitType.OR, BitType.XOR):
                    if not all(isinstance(op, Bit) for op in node.operands):
                        raise TypeError("Boolean operations require Bit operands")
                elif not (
                    all(isinstance(op, Bit) for op in node.operands)
                    or all(isinstance(op, UInt) for op in node.operands)
                ):
                    raise TypeError("comparisons require two Bits or two UInts")
        elif isinstance(node, UInt):
            if not isinstance(node.uint_type, UIntType):
                raise TypeError("A UInt node's uint_type must be a UIntType member")
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
                raise ValueError("UInt bits must match its width")
            if node.uint_type == UIntType.CONST:
                if not all(
                    isinstance(bit, Bit) and bit.bit_type == BitType.CONST
                    for bit in node.bits
                ):
                    raise TypeError("constant UInt requires constant Bit nodes")
            elif node.uint_type == UIntType.BITS:
                if not all(isinstance(bit, Bit) for bit in node.bits):
                    raise TypeError("UInt bits must be Bit nodes")
            else:
                # Owned bits point back to this UInt. Check their metadata here,
                # without following that back-reference as a dependency cycle.
                for index, bit in enumerate(node.bits):
                    if (
                        not isinstance(bit, Bit)
                        or not isinstance(bit.bit_type, BitType)
                        or bit.bit_type != BitType.UINT
                    ):
                        raise ValueError("UInt must own indexed Bit nodes")
                    if type(bit.index) is not int or bit.index != index:
                        raise ValueError("owned bit indices must match their positions")
                    if bit.parent_uint is not node:
                        raise ValueError("owned bit must reference its parent UInt")
                    if not isinstance(bit.operands, tuple) or bit.operands:
                        raise ValueError("owned bit expects no operands")
            if not all(isinstance(op, UInt) for op in node.operands):
                raise TypeError("UInt operations require UInt operands")

        children = node.children
        if not isinstance(children, tuple): 
            raise TypeError("node children must be a tuple")
        visiting.add(node_id)
        for child in children: 
            visit(child)
        # validate individual bits of a UInt CONST separately
        if isinstance(node, UInt) and node.uint_type == UIntType.CONST: 
            for bit in node.bits: 
                visit(bit)
        
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


