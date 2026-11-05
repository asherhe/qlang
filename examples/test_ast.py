import qlang
from qlang.expression import ASTNode, Bit, UInt, BitType, UIntType, print_ast
from qlang.optimize import optimize_expr

def main():
    a = qlang.uint("a", 2)
    b = qlang.uint("b", 2)

    print_ast(a)
    print_ast(b)
    print()

    c = a + b
    print_ast(c)
    print()

    print_ast(optimize_expr(c * 5))

    d = c == 3
    print_ast(d)
    print()

    big_expr = (~~~(d | ~d) ^ ((a<<1)>>1 >= 2)) | ~((b==b) | ~(4>a))
    print_ast(big_expr)
    big_expr_opt = qlang.optimize_expr(big_expr)
    print("optimize_expr:")
    print_ast(big_expr_opt)
    print()

if __name__ == "__main__":
    main()
