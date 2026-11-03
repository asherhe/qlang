from typing import cast

from .expression import ASTNode, Bit, BitType, UInt, UIntType


def _get_type(node: ASTNode): 
    if isinstance(node, Bit): 
        return node.bit_type
    if isinstance(node, UInt): 
        return node.uint_type
    return NotImplemented

# assumes that types not CONST or VAR are valid to recurse through
def _var_checker(
    nodes: tuple[ASTNode, ...],
    assignments: dict[str, int | bool],
) -> None: 
    for node in nodes: 
        type = _get_type(node)
        if type in (BitType.VAR, UIntType.VAR): 
            if node.name not in assignments: 
                raise ValueError(f"variable '{node.name}' is not assigned")
            value = assignments[node.name]
            if isinstance(node, UInt) or (isinstance(node, Bit) and node.index is not None): 
                if not isinstance(value, (int, bool)): 
                    raise ValueError(f"assignment value '{node.name}' must be an integer")
                if value < 0: 
                    raise ValueError(f"assignment value '{node.name}' must be non-negative")
                elif isinstance(node, UInt) and value >= (1 << node.width): 
                    raise ValueError(f"assignment value '{node.name}' cannot be larger than width")
                elif isinstance(node, Bit) and value >= (1 << cast(int, node.parent_width)): 
                    if not node.parent_width: 
                        raise ValueError("parent width missing for Bit belonging to UInt")
                    if value >= (1 << node.parent_width): 
                        raise ValueError(f"assignment value '{node.name}' cannot be larger than parent width")
        elif isinstance(node, Bit) and node.bit_type == BitType.CONST: 
            if node.value is None: 
                raise ValueError("const Bit has missing value")
        elif isinstance(node, UInt) and node.uint_type == UIntType.CONST: 
            if not node.bits: 
                raise ValueError("const UInt has missing bits")
            if not all(b.bit_type == BitType.CONST for b in node.bits):
                raise ValueError("const UInt cannot contain var Bit")
            if not all(isinstance(b.value, bool) for b in node.bits): 
                raise ValueError("const Bit as part of const UInt has missing value")

# prereq: assignments are available for VARs 
def _get_values(
    nodes: tuple[ASTNode, ...], 
    assignments: dict[str, int | bool],
) -> tuple[int | bool, ...]: 
    values = ()
    for node in nodes: 
        type = _get_type(node)
        if isinstance(node, Bit):       # not actually that necessary but for intellisense polymorphism
            if type == BitType.VAR:
                value = assignments[cast(str, node.name)]
                if node.index is not None: 
                    value = bool((value >> node.index) & 1)
                else: 
                    value = bool(value)
                values += (value,)
            elif type == BitType.CONST: 
                values += (cast(bool, node.value),)
            else: 
                return NotImplemented
        elif isinstance(node, UInt): 
            if type == UIntType.VAR: 
                values += (assignments[cast(str, node.name)],)
            elif type == UIntType.CONST: 
                values += (sum(cast(bool, b.value) * (1<<i) for i, b in enumerate(node.bits)),)
            else: 
                return NotImplemented
        else: 
            return NotImplemented
    return values

# XOR, AND, OR
def _bitwise_only_op(
    node: ASTNode, 
    assignments: dict[str, int | bool],
    memo: dict[int, int | bool],
): 
    if not isinstance(node, Bit): 
        raise ValueError("this op cannot be applied to a type other than Bit")
    type = node.bit_type
    match type: 
        case BitType.XOR: 
            name = "XOR"
        case BitType.AND: 
            name = "AND"
        case BitType.OR: 
            name = "OR"
        case _: 
            return NotImplemented
    b1, b2 = node.operands[0], node.operands[1]   # not node.operands[:2] because intellisense can't do polymorphism
    if not isinstance(b1, Bit) or not isinstance(b2, Bit): 
        raise ValueError(f"{name} cannot be applied to a type other than Bit")
    _var_checker((b1, b2), assignments)
    if b1.bit_type not in (BitType.VAR, BitType.CONST): 
        b1v = _evaluate(b1, assignments, memo)
    else: 
        b1v = _get_values((b1,), assignments)[0]
    if b2.bit_type not in (BitType.VAR, BitType.CONST): 
        b2v = _evaluate(b2, assignments, memo)
    else: 
        b2v = _get_values((b2,), assignments)[0]
    match type: 
        case BitType.XOR: 
            return b1v ^ b2v
        case BitType.AND: 
            return b1v & b2v
        case BitType.OR: 
            return b1v | b2v

# EQ, NE, GT, LE, GE, LE
def _comparison_op(
    node: ASTNode, 
    assignments: dict[str, int | bool],
    memo: dict[int, int | bool],
): 
    if not isinstance(node, Bit): 
        raise ValueError("this op cannot be applied to a type other than Bit")
    type = node.bit_type
    u1, u2 = node.operands[0], node.operands[1]
    _var_checker((u1, u2), assignments)
    if _get_type(u1) not in (BitType.VAR, BitType.CONST, UIntType.VAR, UIntType.CONST):
        u1v = _evaluate(u1, assignments, memo)
    else: 
        u1v = _get_values((u1,), assignments)[0]
    if _get_type(u2) not in (BitType.VAR, BitType.CONST, UIntType.VAR, UIntType.CONST): 
        u2v = _evaluate(u2, assignments, memo)
    else: 
        u2v = _get_values((u2,), assignments)[0]
    match type: 
        case BitType.EQ: 
            return u1v == u2v
        case BitType.NE: 
            return u1v != u2v
        case BitType.GT: 
            return u1v > u2v
        case BitType.LT: 
            return u1v < u2v
        case BitType.GE: 
            return u1v >= u2v
        case BitType.LE: 
            return u1v <= u2v
        case _: 
            return NotImplemented

# ADD, MUL, LSHIFT, RSHIFT
def _int_only_op(
    node: ASTNode, 
    assignments: dict[str, int | bool],
    memo: dict[int, int | bool],
): 
    if not isinstance(node, UInt): 
        raise ValueError("this op cannot be applied to a type other than UInt")
    type = node.uint_type
    match type: 
        case UIntType.ADD: 
            name = "ADD"
        case UIntType.MUL: 
            name = "MUL"
        case UIntType.LSHIFT: 
            name = "LSHIFT"
        case UIntType.RSHIFT: 
            name = "RSHIFT"
        case _: 
            return NotImplemented
    u1, u2 = node.operands[0], node.operands[1]
    if not isinstance(u1, UInt) or not isinstance(u2, UInt): 
        raise ValueError(f"{name} cannot be applied to a type other than UInt")
    _var_checker((u1, u2), assignments)
    if _get_type(u1) not in (UIntType.VAR, UIntType.CONST):
        u1v = _evaluate(u1, assignments, memo)
    else: 
        u1v = _get_values((u1,), assignments)[0]
    if _get_type(u2) not in (UIntType.VAR, UIntType.CONST): 
        u2v = _evaluate(u2, assignments, memo)
    else: 
        u2v = _get_values((u2,), assignments)[0]
    match type: 
        case UIntType.ADD: 
            return u1v + u2v
        case UIntType.MUL: 
            return u1v * u2v
        case UIntType.LSHIFT: 
            if u2v >= node.width: 
                return 0
            return (u1v << u2v) & ((1 << node.width) - 1)  # assumes fixed width for uint
        case UIntType.RSHIFT: 
            return u1v >> u2v

def _evaluate_uncached(
    node: ASTNode, 
    assignments: dict[str, int | bool],
    memo: dict[int, int | bool],
) -> int | bool: 
    if not node.operands:  # leaf node
        _var_checker((node,), assignments)
        if _get_type(node) not in (BitType.CONST, BitType.VAR, UIntType.CONST, UIntType.VAR): 
            raise ValueError("operator node must have operands")
        return _get_values((node,), assignments)[0]
    else:                  # operators
        if isinstance(node, Bit): 
            type = node.bit_type
            if type == BitType.NOT: 
                if len(node.operands) != 1: 
                    raise ValueError("NOT must have exactly one operand")
                b = node.operands[0]
                if not isinstance(b, Bit): 
                    raise ValueError("NOT cannot be applied to a type other than Bit")
                _var_checker((b,), assignments)
                if b.bit_type in (BitType.VAR, BitType.CONST): 
                    return not _get_values((b,), assignments)[0]
                return not _evaluate(b, assignments, memo)
            if len(node.operands) != 2: 
                raise ValueError("binary operations must have exactly two operands")
            if type in (BitType.XOR, BitType.AND, BitType.OR): 
                return _bitwise_only_op(node, assignments, memo)
            if type in (BitType.EQ, BitType.NE, BitType.GT, BitType.LT, BitType.GE, BitType.LE): 
                return _comparison_op(node, assignments, memo)
            raise ValueError("BitType not recognized")
        if isinstance(node, UInt): 
            if len(node.operands) != 2: 
                raise ValueError("binary operations must have exactly two operands")
            if node.uint_type not in (UIntType.ADD, UIntType.MUL, UIntType.LSHIFT, UIntType.RSHIFT): 
                raise ValueError("UIntType not recognized")
            return _int_only_op(node, assignments, memo)
        raise ValueError("ASTNode not recognized")

# helper that manages caching
def _evaluate(node, assignments, memo): 
    key = id(node)
    if key in memo: 
        return memo[key]
    memo[key] = _evaluate_uncached(node, assignments, memo)
    return memo[key]

# the public interface
def evaluate(node, assignments): 
    memo = {}
    return _evaluate(node, assignments, memo)
