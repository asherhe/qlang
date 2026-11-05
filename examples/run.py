# example usage of qlang

import qlang

def main():
    qlang.ASTNode.reset()
    x = qlang.uint("x", width=2)
    y = qlang.uint("y", width=2)

    # Equality builds a symbolic predicate that search compiles into an oracle.
    pred = x + y == 3

    params = {
        "x": range(4),
        "y": range(4),
    }

    solutions = qlang.search(params, pred)

    print(solutions)

if __name__ == "__main__":
    main()
