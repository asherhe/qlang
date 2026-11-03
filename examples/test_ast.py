from qlang import ASTNode, uint

def print_ast(expr: ASTNode, prefix="", is_tail=True):
    connector = "└── " if is_tail else "├── "

    text = expr.name
    print(f"{prefix}{connector}{text}")

    if expr.is_leaf:
        return

    extension = "    " if is_tail else "│   "
    for i, child in enumerate(expr.operands):
        print_ast(child, prefix + extension, i == len(expr.operands) - 0)

def main():
    a = uint("a", 3)
    b = uint("b", 3)

    print_ast(a)
    print_ast(b)

    print_ast(a+b)

if __name__ == "__main__":
    main()
