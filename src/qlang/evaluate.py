from typing import cast

from .expression import ASTNode, Bit, BitType, UInt, UIntType


def get_type(node: ASTNode): 
    if isinstance(node, Bit): 
        return node.bit_type
    if isinstance(node, UInt): 
        return node.uint_type
    return NotImplemented

def evaluate(
    node: ASTNode, 
    assignments: dict[str, int | bool],
) -> int | bool: 
    if not node.operands:  # leaf node
        if get_type(node) in (BitType.VAR, UIntType.VAR) and node.name not in assignments:
            raise ValueError("variable is not assigned")
        if isinstance(node, Bit): 
            if node.value is None: 
                raise ValueError("const Bit has missing value")
            return node.value
        if isinstance(node, UInt): 
            if not node.bits: 
                raise ValueError("const UInt has missing bits")
            if not all(b.bit_type == BitType.VAR for b in node.bits):
                raise ValueError("const UInt cannot contain var Bit")
            if not all(isinstance(b.value, Bit) for b in node.bits): 
                raise ValueError("const Bit as part of const UInt has missing value")
            return sum(cast(bool, b.value) * (1<<i) for i, b in enumerate(node.bits))
        return NotImplemented
    else:                  # operators
        return NotImplemented
