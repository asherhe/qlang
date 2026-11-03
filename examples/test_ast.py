import qlang
from qlang.expression import ASTNode, Bit, UInt, BitType, UIntType

def print_ast(expr: ASTNode, prefix="", is_tail=True):
    connector = "└─" if is_tail else "├─"

    text = ""
    if isinstance(expr, Bit):
        if expr.bit_type == BitType.CONST:
            text = f"Bit {1 if expr.value else 0}"
        elif expr.bit_type == BitType.VAR:
            idx_str = f"[{expr.index}]" if expr.index else ""
            text = f"Bit {expr.name}{idx_str}"
        else:
            BIT_OPS = {
                BitType.NOT: "~",
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
            text = f"Bit op{BIT_OPS[expr.bit_type]}"
    elif isinstance(expr, UInt):
        if expr.uint_type == UIntType.CONST:
            text = f"UInt {expr.const_value}"
        elif expr.uint_type == UIntType.VAR:
            text = f"UInt {expr.name}"
        else:
            UINT_OPS = {
                UIntType.ADD: "+",
                UIntType.MUL: "*",
                UIntType.LSHIFT: "<<",
                UIntType.RSHIFT: ">>",
            }
            text = f"UInt op{UINT_OPS[expr.uint_type]}"
        text = f"{text} ({expr.width}-bit)"

    print(f"{prefix}{connector}{text}")

    if expr.is_leaf:
        return

    extension = "  " if is_tail else "│ "
    for i, child in enumerate(expr.operands):
        print_ast(child, prefix + extension, i == len(expr.operands) - 1)

def main():
    a = qlang.uint("a", 2)
    b = qlang.uint("b", 2)

    print_ast(a)
    print_ast(b)
    print()

    c = a + b
    print_ast(c)
    print()

    d = c == 3
    print_ast(d)
    print()

    big_expr = (~~~(d | ~d) ^ ((a<<1)>>1 >= 2)) | ~((b==b) | ~(4>a))
    print_ast(big_expr)
    big_expr_decomp = qlang.decompose_ops(big_expr)
    print("decompose_ops:")
    print_ast(big_expr_decomp)
    big_expr_opt = qlang.optimize_expr(big_expr_decomp)
    print("optimize_expr:")
    print_ast(big_expr_opt)
    print()

if __name__ == "__main__":
    main()
