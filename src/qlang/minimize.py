from .expression import ASTNode, Bit, UInt, BitType, UIntType, node_type
from .evaluate import evaluate

from math import pi
from itertools import product
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
            left, right = expr.operands
            left_const, left_coeffs = extract_linear(left)
            right_const, right_coeffs = extract_linear(right)
            const = left_const + right_const
            coeffs = {}
            for key, value in left_coeffs.items(): 
                if key not in right_coeffs: 
                    coeffs[key] = value
                else: 
                    coeffs[key] = value + right_coeffs[key]
            return const, coeffs
        if type == UIntType.MUL: 
            left, right = expr.operands
            left_const, left_coeffs = extract_linear(left)
            right_const, right_coeffs = extract_linear(right)
            if left_coeffs and right_coeffs: 
                return NotImplemented
            const = left_const * right_const
            coeffs = {}
            if not left_coeffs: 
                coeffs = {key: value * left_const for key, value in right_coeffs.items()}
            else: 
                coeffs = {key: value * right_const for key, value in left_coeffs.items()}
            return const, coeffs
    return NotImplemented

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

    # if evaluate_linear(constraint) is not NotImplemented: 
    #     ...

    operator = constraint_to_cost_op(constraint, widths, objective, penalty)
    state_at, mean_cost = qaoa_creator(operator)
    best_angles = min(product(grid, repeat=2), key=mean_cost)

    counts = state_at(*best_angles).sample_counts(1024)
    observed = [decode_state(int(bits, 2), widths) for bits in counts]
    feasible = (assignment for assignment in observed if evaluate(constraint, assignment))
    if objective is None: 
        return next(feasible, None)
    return min(feasible, key=objective, default=None)
