"""Build, evaluate, simplify, and compile a small symbolic problem."""

import qlang


def main():
    qlang.ASTNode.reset()
    x = qlang.uint("x", width=2)
    y = qlang.uint("y", width=2)
    assignment = {"x": 1, "y": 2}

    constraint = x + y == 3
    objective = 2 * x + y
    print("Assignment:", assignment)
    print("Constraint satisfied:", qlang.evaluate(constraint, assignment))
    print("Objective cost:", qlang.evaluate(objective, assignment))

    simplified = qlang.optimize_expr(x + 0)
    assert qlang.evaluate(simplified, assignment) == 1
    print("Simplified expression:", simplified)

    oracle = qlang.compile_oracle(constraint)
    print("Oracle input qubits:", oracle.inputs)
    print("Oracle total qubits:", oracle.circuit.num_qubits)
    print("Oracle ancillas:", oracle.total_ancillas)


if __name__ == "__main__":
    main()
