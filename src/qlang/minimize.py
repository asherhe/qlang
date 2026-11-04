from .expression import ASTNode, Bit, UInt, var_bit, bits_uint
from .compile import Oracle, compile_oracle, OutputMode
from .evaluate import evaluate

import numpy as np
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
beta = parameters["β[0]"]
gamma = parameters["γ[0]"]

a, b = var_bit("a"), var_bit("b")

clash = ~(a ^ b)
cost = 3 * bits_uint((clash,)) + bits_uint((a,))