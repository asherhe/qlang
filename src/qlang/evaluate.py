from typing import cast

from .expression import ASTNode, Bit, BitType, UInt, UIntType


def get_type(node: ASTNode): 
    if isinstance(node, Bit): 
        return node.bit_type
    if isinstance(node, UInt): 
        return node.uint_type
    return NotImplemented

def var_checker(
    nodes: tuple[ASTNode, ...],
    assignments: dict[str, int | bool],
) -> None: 
    for node in nodes: 
        if get_type(node) in (BitType.VAR, UIntType.VAR): 
            if node.name not in assignments: 
                raise ValueError(f"variable '{node.name}' is not assigned")
        elif isinstance(node, Bit) and node.bit_type == BitType.CONST: 
            if node.value is None: 
                raise ValueError("const Bit has missing value")
        elif isinstance(node, UInt) and node.uint_type == UIntType.CONST: 
            if not node.bits: 
                raise ValueError("const UInt has missing bits")
            if not all(b.bit_type == BitType.VAR for b in node.bits):
                raise ValueError("const UInt cannot contain var Bit")
            if not all(isinstance(b.value, Bit) for b in node.bits): 
                raise ValueError("const Bit as part of const UInt has missing value")
        else: 
            raise NotImplementedError("node type not implemented yet")

# prereq: assignments are available for VARs 
def get_values(
    nodes: tuple[ASTNode, ...], 
    assignments: dict[str, int | bool],
) -> tuple[int | bool, ...]: 
    values = ()
    for node in nodes: 
        type = get_type(node)
        if isinstance(node, Bit):       # not actually that necessary but for intellisense polymorphism
            if type == BitType.VAR:
                values += (assignments[cast(str, node.name)],)
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
def bitwise_only_op(
    node: ASTNode, 
    assignments: dict[str, int | bool],
    type: BitType,
): 
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
    var_checker((b1, b2), assignments)
    if b1.bit_type not in (BitType.VAR, BitType.CONST) or b2.bit_type not in (BitType.VAR, BitType.CONST): 
        b1v, b2v = evaluate(b1, assignments), evaluate(b2, assignments)
    else: 
        b1v, b2v = get_values((b1, b2), assignments)
    match type: 
        case BitType.XOR: 
            return b1v ^ b2v
        case BitType.AND: 
            return b1v & b2v
        case BitType.OR: 
            return b1v | b2v

# EQ, NE, GT, LE, GE, LE
def comparison_op(
    node: ASTNode, 
    assignments: dict[str, int | bool],
    type: BitType,
): 
    u1, u2 = node.operands[0], node.operands[1]
    var_checker((u1, u2), assignments)
    if (get_type(u1) not in (BitType.VAR, BitType.CONST, UIntType.VAR, UIntType.CONST) or
        get_type(u2) not in (BitType.VAR, BitType.CONST, UIntType.VAR, UIntType.CONST)): 
        u1v, u2v = evaluate(u1, assignments), evaluate(u2, assignments)
    else: 
        u1v, u2v = get_values((u1, u2), assignments)
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
def int_only_op(
    node: ASTNode, 
    assignments: dict[str, int | bool],
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
    var_checker((u1, u2), assignments)
    if get_type(u1) not in (UIntType.VAR, UIntType.CONST) or get_type(u2) not in (UIntType.VAR, UIntType.CONST):
        u1v, u2v = evaluate(u1, assignments), evaluate(u2, assignments)
    else: 
        u1v, u2v = get_values((u1, u2), assignments)
    match type: 
        case UIntType.ADD: 
            return u1v + u2v
        case UIntType.MUL: 
            return u1v * u2v
        case UIntType.LSHIFT: 
            return u1v << u2v
        case UIntType.RSHIFT: 
            return u1v >> u2v

def evaluate(
    node: ASTNode, 
    assignments: dict[str, int | bool],
) -> int | bool: 
    if not node.operands:  # leaf node
        var_checker((node,), assignments)
        return get_values((node,), assignments)[0]
    else:                  # operators
        if isinstance(node, Bit): 
            match node.bit_type: 
                case BitType.NOT: 
                    b = node.operands[0]
                    if not isinstance(b, Bit): 
                        raise ValueError("NOT cannot be applied to a type other than Bit")
                    var_checker((b,), assignments)
                    if b.bit_type in (BitType.VAR, BitType.CONST): 
                        return not get_values((node,), assignments)[0]
                    return not evaluate(b, assignments)
                case BitType.XOR: 
                    return bitwise_only_op(node, assignments, BitType.XOR)
                case BitType.AND: 
                    return bitwise_only_op(node, assignments, BitType.AND)
                case BitType.OR: 
                    return bitwise_only_op(node, assignments, BitType.OR)
                case BitType.EQ:
                    return comparison_op(node, assignments, BitType.EQ)
                case BitType.NE:
                    return comparison_op(node, assignments, BitType.NE)
                case BitType.GT:
                    return comparison_op(node, assignments, BitType.GT)
                case BitType.LT:
                    return comparison_op(node, assignments, BitType.LT)
                case BitType.GE:
                    return comparison_op(node, assignments, BitType.GE)
                case BitType.LE:
                    return comparison_op(node, assignments, BitType.LE)
                case _: 
                    return NotImplemented
        if isinstance(node, UInt): 
            if node.uint_type not in (UIntType.ADD, UIntType.MUL, UIntType.LSHIFT, UIntType.RSHIFT): 
                return NotImplemented
            return int_only_op(node, assignments)
        return NotImplemented
