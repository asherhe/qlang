"""Minimize a symbolic objective subject to a linear equality using QAOA."""

import qlang


def main():
    # Start a new expression namespace before constructing the problem.
    qlang.ASTNode.reset()
    x = qlang.uint("x", width=2)
    y = qlang.uint("y", width=2)

    constraint = x + y == 3
    objective = 2 * x + y

    # The objective ranges from 0 to 9. A penalty of 10 makes any unit
    # violation of the integer equality cost more than that full range.
    solution = qlang.qaoa(
        [x, y],
        constraint,
        objective=objective,
        penalty=10.0,
        iterations=16,
    )

    if solution is None:
        print("No feasible assignment was sampled. Try another run.")
        return

    assert qlang.evaluate(constraint, solution)
    print("QAOA assignment:", solution)
    print("Constraint satisfied:", qlang.evaluate(constraint, solution))
    print("Objective cost:", qlang.evaluate(objective, solution))
    # Under x + y == 3, the objective is 3 + x. Its exact minimum is 3.
    print("Known minimum cost:", 3)


if __name__ == "__main__":
    main()
