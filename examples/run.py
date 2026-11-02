# example usage of qlang

import qlang

def main():
    x = qlang.uint("x", width=2)
    y = qlang.uint("y", width=2)

    # note: overloading == is cooked because it interferes with other language equality stuff
    # probably make a .eq() operator instead for the equality operator
    pred = x + y == 3

    params = {
        "x": range(4),
        "y": range(4),
    }

    solutions = qlang.search(params, pred)

    print(solutions)

if __name__ == "__main__":
    main()
