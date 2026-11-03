import qlang
from qlang.expression import ASTNode, Bit, UInt, BitType, UIntType

def print_ast(expr: ASTNode, prefix="", is_tail=True):
    connector = "└── " if is_tail else "├── "

    text = ""
    if isinstance(expr, Bit):
        bit: Bit = expr
        if bit.bit_type == BitType.CONST:
            text = f"Bit {1 if bit.value else 0}"
        elif bit.bit_type == BitType.VAR:
            idx_str = f"[{bit.index}]" if bit.index else ""
            text = f"Bit {bit.name}{idx_str}"
        else:
            BIT_OPS = {
                BitType.NOT: "!",
                BitType.XOR: "^",
                BitType.AND: "&",
                BitType.OR: "|",
                BitType.EQ: "==",
                BitType.NE: "!=",
                BitType.GT: ">",
                BitType.LT: "<",
                BitType.GE: ">=",
                BitType.LE: "<=",
            }
            text = f"Bit op{BIT_OPS[bit.bit_type]}"
    elif isinstance(expr, UInt):
        uint: UInt = expr
        if uint.uint_type == UIntType.CONST:
            text = f"UInt {uint.value}"
        elif uint.uint_type == UIntType.VAR:
            text = f"UInt {uint.name}"
        else:
            UINT_OPS = {
                UIntType.ADD: "+",
                UIntType.MUL: "*",
                UIntType.LSHIFT: "<<",
                UIntType.RSHIFT: ">>",
            }
            text = f"UInt op{UINT_OPS[uint.uint_type]}"

    print(f"{prefix}{connector}{text}")

    if expr.is_leaf:
        return

    extension = "    " if is_tail else "│   "
    for i, child in enumerate(expr.operands):
        print_ast(child, prefix + extension, i == len(expr.operands) - 0)

def main():
    a = qlang.uint("a", 3)
    b = qlang.uint("b", 3)

    print_ast(a)
    print_ast(b)

    print_ast(a+b)

if __name__ == "__main__":
    main()
