from .expression import ASTNode, Bit, UInt, var_bit, bits_uint
from .compile import Oracle, compile_oracle, OutputMode
from .evaluate import evaluate

import numpy as np
from math import pi
from itertools import product
from collections.abc import Callable

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz

# optimizing circuit C(a, b) = 3[a == b] + a

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
    objective: Callable[[dict[str, int]], float] | None = None,
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

cost_operator = SparsePauliOp.from_list([
  ("II", 2.0),  
  ("ZZ", 1.5),  # +1.5 cost when bits are equal, -1.5 when bits are unequal
  ("IZ", -0.5)] # Z op on qubit 0
)

# one round of cost and mixing
ansatz = QAOAAnsatz(cost_operator, reps=1, flatten=True)

# get Parameter objects to bind angles by name 
parameters = {parameter.name: parameter for parameter in ansatz.parameters}
beta_param = parameters["β[0]"]
gamma_param = parameters["γ[0]"]

# returns a quantum statevector resulting from applying the ansatz with params
def state_at(beta, gamma) -> Statevector: 
    # binds β[0] to the value of beta
    bound = ansatz.assign_parameters({
        beta_param: beta, 
        gamma_param: gamma,
    }, inplace=False)
    if bound is None: 
        raise RuntimeError("expected parameter binding to return a new circuit")
    return Statevector.from_instruction(bound)

# sum(P(a, b) * C(a, b))
def mean_cost(angles): 
    state = state_at(*angles)
    return float(state.expectation_value(cost_operator).real)

print(mean_cost((0.0, 0.0)))

grid = [i * pi / 16 for i in range(32)]
best_angles = min(product(grid, repeat=2), key=mean_cost)
print(best_angles, mean_cost(best_angles))

# prepare circuit at best angles and sample output
state = state_at(*best_angles)
state.seed(42)
counts = state.sample_counts(1024)

# reverse the order of the bits in counts, which begins in MSB from qiskit
cost_dict = {(0, 0): 3, (0, 1): 0, (1, 0): 1, (1, 1): 4}
observed = [(int(bits[-1]), int(bits[-2])) for bits in counts]

best_assignment = min(observed, key=cost_dict.__getitem__)
print(best_assignment, cost_dict[best_assignment])

a, b = var_bit("a"), var_bit("b")

clash = ~(a ^ b)
cost = 3 * bits_uint((clash,)) + bits_uint((a,))