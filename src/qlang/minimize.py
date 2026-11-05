from .expression import ASTNode, Bit, UInt, BitType, UIntType, node_type
from .evaluate import evaluate

from math import pi
from itertools import product, combinations
from collections import defaultdict
from collections.abc import Callable
from typing import cast

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz

# converts a quantum state into decipherable values
# based on order of qubits and their widths
def decode_state(state: int, widths: dict[str, int]) -> dict[str, int]: 
    assignment = {}
    for name, width in widths.items(): 
        assignment[name] = state & ((1 << width) - 1)
        state >>= width
    return assignment

# converts a constraint equation into a cost operator that QAOAAnsatz can read
# objective can be omitted if the problem is feasibility rather than optimization
def constraint_to_cost_op(
    constraint: Bit, 
    widths: dict[str, int], 
    objective: Callable[[dict[str, int]], float] | None = None,  # a lower result is preferred
    penalty: float = 1.0,
) -> SparsePauliOp: 
    num_qubits = sum(widths.values())
    basis_size = 1 << num_qubits

    # cost of each of the possible 2^n states
    energies = []
    for state in range(basis_size): 
        assignment = decode_state(state, widths)
        violation = 0 if evaluate(constraint, assignment) else 1
        base_cost = 0.0 if objective is None else objective(assignment)
        energies.append(base_cost + penalty * violation)

    terms = []
    # 1 bit set in mask means to try Z on this qubit, 0 bit means I
    for mask in range(basis_size): 
        # determines the coefficient of each combination of Pauli operators 
        # necessary for the cost Hamiltonian
        coefficient = sum(
            (-1 if (index & mask).bit_count() % 2 else 1) * energy
            for index, energy in enumerate(energies)
        ) / basis_size
        if coefficient != 0.0: 
            label = "".join(
                "Z" if mask & (1 << qubit) else "I"
                for qubit in reversed(range(num_qubits))
            )
            terms.append((label, coefficient))
    if not terms: 
        terms.append(("I" * num_qubits, 0.0))
    return SparsePauliOp.from_list(terms)

def qaoa_creator(cost_operator: SparsePauliOp): 
    # one round of cost and mixing
    ansatz = QAOAAnsatz(cost_operator, reps=1, flatten=True)

    # get Parameter objects to bind angles by name 
    parameters = {parameter.name: parameter for parameter in ansatz.parameters}
    beta_param = parameters["β[0]"]
    gamma_param = parameters.get("γ[0]")

    # returns a quantum statevector resulting from applying the ansatz with params
    def state_at(beta, gamma) -> Statevector: 
        # binds β[0] to the value of beta
        bindings = {beta_param: beta}
        if gamma_param is not None: 
            bindings[gamma_param] = gamma
        bound = ansatz.assign_parameters(bindings, inplace=False)
        if bound is None: 
            raise RuntimeError("expected parameter binding to return a new circuit")
        return Statevector.from_instruction(bound)

    # sum(P(a, b) * C(a, b))
    def mean_cost(angles): 
        state = state_at(*angles)
        return float(state.expectation_value(cost_operator).real)

    return state_at, mean_cost

# guards against results that are NotImplemented in recursion
def extract_checker(expr: ASTNode): 
    left, right = expr.operands
    left_res, right_res = extract_linear(left), extract_linear(right)
    if left_res is NotImplemented or right_res is NotImplemented: 
        return NotImplemented            
    left_const, left_coeffs = left_res
    right_const, right_coeffs = right_res
    return left_const, left_coeffs, right_const, right_coeffs

# extracts linear expressions to the left and right of the equals sign
def extract_linear(expr: ASTNode) -> tuple[int | bool, dict[str, int]]: 
    type = node_type(expr)
    if isinstance(expr, Bit): 
        if type == BitType.CONST: 
            return cast(bool, expr.value), {}
        if type == BitType.VAR: 
            return 0, {cast(str, expr.name): 1}
        return NotImplemented
    if isinstance(expr, UInt): 
        if type == UIntType.CONST: 
            return expr.const_value, {}
        if type == UIntType.VAR: 
            return 0, {cast(str, expr.name): 1}
        if type == UIntType.ADD: 
            res = extract_checker(expr)
            if res is NotImplemented: 
                return NotImplemented
            left_const, left_coeffs, right_const, right_coeffs = res
            const = left_const + right_const
            coeffs = left_coeffs.copy()
            for name, coeff in right_coeffs.items(): 
                coeffs[name] = coeffs.get(name, 0) + coeff
            return const, coeffs
        if type == UIntType.MUL: 
            res = extract_checker(expr)
            if res is NotImplemented: 
                return NotImplemented
            left_const, left_coeffs, right_const, right_coeffs = res
            const = left_const * right_const
            coeffs = {}
            if not left_coeffs: 
                coeffs = {key: value * left_const for key, value in right_coeffs.items() if value * left_const != 0}
            else: 
                coeffs = {key: value * right_const for key, value in left_coeffs.items() if value * right_const != 0}
            return const, coeffs
    return NotImplemented

# evaluate a constraint expression to see if it is linear, and move variables and constants to one side 
def evaluate_linear(expr: Bit) -> tuple[int | bool, dict[str, int]]: 
    if len(expr.operands) == 2 and expr.bit_type == BitType.EQ: 
        res = extract_checker(expr)
        if res is NotImplemented: 
            return NotImplemented
        left_const, left_coeffs, right_const, right_coeffs = res
        const = left_const - right_const
        coeffs = left_coeffs.copy()
        for name, coeff in right_coeffs.items(): 
            coeffs[name] = coeffs.get(name, 0) - coeff
        return const, coeffs
    return NotImplemented

# expands a list of coefficient assignments to a bit representation of each variable
# e.g. 3x of width 2 becomes 3x[0] + 6x[1]
def expand_to_bits(
    coefficients: dict[str, int], 
    widths: dict[str, int],
) -> dict[int, float]: 
    weights = {}
    offset = 0
    for name, width in widths.items(): 
        coeff = coefficients.get(name, 0)
        for idx in range(width): 
            qubit = offset + idx
            if coeff != 0: 
                weights[qubit] = coeff * (1 << idx)
        offset += widths[name]
    return weights

# calculates Pauli coefficients for a linear constraint
# 0 means identity, 1 << i means Zi, (1 << i) | (i << j) means ZiZj
def squared_residuals(
    constant: float, 
    weights: dict[int, float], 
    penalty: float,
) -> dict[int, float]: 
    terms = defaultdict(float)
    a = constant + sum(weights.values()) / 2
    terms[0] += penalty * a * a + sum(w * w for w in weights.values()) / 4
    for q1, w1 in weights.items(): 
        terms[1 << q1] += -penalty * a * w1
        for q2, w2 in weights.items(): 
            if q1 == q2: 
                continue
            terms[(1 << q1) | (1 << q2)] += penalty * w1 * w2 / 2
    return terms

# calculates Pauli coefficients for an objective applied linearly
def linear_objective_terms(constant: float, weights: dict[int, float]) -> dict[int, float]: 
    terms = {0: constant + sum(val for val in weights.values()) / 2}
    for q, w in weights.items(): 
        terms[1 << q] = -w / 2
    return terms

# converts linear terms to Pauli operators to feed into QAOAAnsatz
def terms_to_op(terms: dict[int, float], num_qubits: int) -> list[tuple[str, int]]: 
    pauli_terms = []
    for mask, coeff in terms.items(): 
        if coeff == 0: 
            continue
        label = "".join(
            "Z" if mask & (1 << qubit) else "I"
            for qubit in reversed(range(num_qubits))
        )
        pauli_terms.append((label, coeff))
    if not pauli_terms: 
        pauli_terms.append(("I" * num_qubits, 0.0))
    return pauli_terms

def qaoa(
    vars: list[Bit | UInt], 
    constraint: Bit, 
    objective: Callable[[dict[str, int]], float] | None = None, 
    penalty: float = 1.0, 
    iterations: int = 32, 
) -> dict[str, int] | None: 
    widths = {}
    for var in vars: 
        if isinstance(var, UInt): 
            widths[var.name] = var.width
        else: 
            widths[var.name] = 1
    grid = [i * 2*pi / iterations for i in range(iterations)]

    linear = evaluate_linear(constraint)
    if linear is not NotImplemented: 
        ...

    operator = constraint_to_cost_op(constraint, widths, objective, penalty)
    state_at, mean_cost = qaoa_creator(operator)
    best_angles = min(product(grid, repeat=2), key=mean_cost)

    counts = state_at(*best_angles).sample_counts(1024)
    observed = [decode_state(int(bits, 2), widths) for bits in counts]
    feasible = (assignment for assignment in observed if evaluate(constraint, assignment))
    if objective is None: 
        return next(feasible, None)
    return min(feasible, key=objective, default=None)
