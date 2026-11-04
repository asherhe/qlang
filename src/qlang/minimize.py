from .expression import ASTNode, Bit, UInt, var_bit, bits_uint
from .compile import Oracle, compile_oracle, OutputMode
from .evaluate import evaluate

import numpy as np
from math import pi
from itertools import product
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz

# optimizing circuit C(a, b) = 3[a == b] + a

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