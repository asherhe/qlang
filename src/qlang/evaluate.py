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

def evaluate(
    node: ASTNode, 
    assignments: dict[str, int | bool],
) -> int | bool: 
    if not node.operands:  # leaf node
        var_checker((node,), assignments)
        if isinstance(node, Bit): 
            return cast(bool, node.value)
        if isinstance(node, UInt): 
            return sum(cast(bool, b.value) * (1<<i) for i, b in enumerate(node.bits))
        return NotImplemented
    else:                  # operators
        if isinstance(node, Bit): 
            match node.bit_type: 
                case BitType.NOT: 
                    if not isinstance(node.operands[0], Bit): 
                        raise ValueError("NOT cannot be applied to a type other than Bit")
                    b = node.operands[0]
                    var_checker((node,), assignments)
                    if b.bit_type == BitType.VAR: 
                        return assignments[cast(str, b.name)]
                    if b.bit_type == BitType.CONST: 
                        return not b.value
                    return not evaluate(b, assignments)
                case _: 
                    return NotImplemented
        if isinstance(node, UInt): 
            return NotImplemented
        return NotImplemented
