from .expression import ASTNode, BitType, UIntType, Bit, UInt, const_bit
from operator import attrgetter
from typing import Callable, cast


# -------- expression optimization --------

Rule = Callable[[ASTNode], ASTNode]

def rebuild(expr: ASTNode, operands: tuple[ASTNode, ...]) -> ASTNode:
    """reconstruct an expression with new operands"""
    # nodes are hash-consed, so unchanged operands mean the node is already canonical
    if all(old is new for old, new in zip(expr.operands, operands)):
        return expr
    if isinstance(expr, Bit):
        return Bit.cache_bit(expr.bit_type, operands)
    if isinstance(expr, UInt):
        return UInt.cache_int(expr.uint_type, expr.width, operands)
    raise TypeError("only Bit and UInt nodes are supported")

def traverse_transform(
    expr: ASTNode,
    rule: Rule,
    memo: dict[int, ASTNode] | None = None,
) -> ASTNode:
    """
    apply transformation rule to every ASTNode of the expression DAG, traversing it bottom-up

    the rule receives a non-leaf node whose operands have already been processed by the rule and returns its replacement.

    execution results are memoized to avoid reprocessing repeat ASTNodes
    """
    if expr.is_leaf:
        return expr

    if memo is None:
        memo = {}
    key = id(expr)
    if key not in memo:
        operands = tuple(traverse_transform(op, rule, memo) for op in expr.operands)
        memo[key] = rule(rebuild(expr, operands))
    return memo[key]

def _decompose_ops(expr: ASTNode) -> ASTNode:
    """
    decompose operations that cannot be directly built with quantum gates into primitive operators
    """
    bit_args = cast(tuple[Bit, ...], expr.operands)
    uint_args = cast(tuple[UInt, ...], expr.operands)

    if isinstance(expr, Bit):
        if expr.bit_type == BitType.OR:
            # decompose into AND, NOT by de morgan's laws

            # assuming only binary OR operation
            # maybe we will do n-ary OR eventually but not now
            expr = ~(~bit_args[0] & ~bit_args[1])
        elif expr.bit_type == BitType.EQ:
            # != is simpler to implement in gates
            expr = ~(uint_args[0] != uint_args[1])
        # collapse all comparison operators to LT to keep compilation clean
        elif expr.bit_type == BitType.GT:
            expr = uint_args[1] < uint_args[0]
        elif expr.bit_type == BitType.LE:
            expr = ~(uint_args[1] < uint_args[0])
        elif expr.bit_type == BitType.GE:
            expr = ~(uint_args[0] < uint_args[1])
            
    elif isinstance(expr, UInt):
        pass

    return expr

def _collapse_identities(expr: ASTNode) -> ASTNode:
    """
    eliminate superfluous expression nodes that are no-ops or can be simplified
    """
    bit_args = cast(tuple[Bit, ...], expr.operands)
    uint_args = cast(tuple[UInt, ...], expr.operands)

    if isinstance(expr, Bit):
        if expr.bit_type == BitType.NOT:
            # fixed values
            if bit_args[0].bit_type == BitType.CONST:
                expr = const_bit(not bit_args[0].value)
            elif bit_args[0].bit_type == BitType.NOT:
                expr = bit_args[0].operands[0]

        elif expr.bit_type == BitType.EQ:
            # skip - assume all EQ have been decomposed
            pass
        elif expr.bit_type == BitType.NE:
            # a != a  ->  false
            if uint_args[0] is uint_args[1]:
                expr = const_bit(False)

        elif expr.bit_type == BitType.LT:
            a, b = uint_args
            atype = a.uint_type
            btype = b.uint_type
            if a is b:
                # a < a  ->  0
                expr = const_bit(False)
            elif atype == UIntType.CONST:
                if btype == UIntType.CONST:
                    expr = const_bit(a.const_value < b.const_value)
            elif btype == UIntType.CONST:
                bval = b.const_value
                if bval == 0:
                    expr = const_bit(False)
                elif bval >= (1 << a.width):
                    expr = const_bit(True)

        elif expr.bit_type == BitType.GT:
            # skip - assume all GT have been decomposed
            pass
        elif expr.bit_type == BitType.LE:
            # skip - assume all LE have been decomposed
            pass
        elif expr.bit_type == BitType.GE:
            # skip - assume all GE have been decomposed
            pass
        else:
            # now all commutative (Bit, Bit) -> Bit operators

            # sort arguments by type
            # places CONST and VAR at the front
            sorted_bits = sorted(bit_args, key=attrgetter("bit_type"))
            a, b = sorted_bits
            atype = a.bit_type
            btype = b.bit_type
            aval = cast(bool, a.value)
            bval = cast(bool, b.value)

            # both constants
            ab_const = atype == btype == BitType.CONST

            # a is ~b
            a_xor_b = (atype == BitType.NOT and a.operands[0] is b
                      or btype == BitType.NOT and b.operands[0] is a)
            if expr.bit_type == BitType.XOR:
                if ab_const:
                    # direct evaluation of const
                    expr = const_bit(aval ^ bval)
                elif atype == BitType.CONST:
                    # 0 ^ b  ->  b
                    # 1 ^ b  ->  ~b
                    expr = ~b if aval else b
                elif a is b:
                    # a ^ a  ->  0
                    expr = const_bit(False)
                elif a_xor_b:
                    # a ^ ~a  ->  1
                    expr = const_bit(True)

            elif expr.bit_type == BitType.AND:
                if ab_const:
                    # direct evaluation of const
                    expr = const_bit(aval & bval)
                elif atype == BitType.CONST:
                    # 0 & b  ->  0
                    # 1 & b  ->  b
                    expr = b if aval else const_bit(False)
                elif a is b:
                    # a & a  ->  a
                    expr = a
                elif a_xor_b:
                    # a & ~a  ->  0
                    expr = const_bit(False)

            elif expr.bit_type == BitType.OR:
                # skip - assume all OR have been decomposed
                pass
            
    elif isinstance(expr, UInt):
        if expr.uint_type == UIntType.ADD:
            pass
        elif expr.uint_type == UIntType.SUB:
            pass
        elif expr.uint_type == UIntType.MUL:
            pass
        elif expr.uint_type == UIntType.LSHIFT:
            pass
        elif expr.uint_type == UIntType.RSHIFT:
            pass

    return expr

def optimize_expr(expr: ASTNode) -> ASTNode:
    """
    apply simplifications to minimize the complexity and gate requirements of the expression tree
    """
    expr = traverse_transform(expr, _decompose_ops)
    expr = traverse_transform(expr, _collapse_identities)
    return expr

