from .expression import ASTNode, Bit, UInt

from collections.abc import Iterable

from .compile import Oracle, compile_oracle, OutputMode

from .evaluate import evaluate

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister, transpile

import math

# compute the optimal k based on the formula
def _optimal_k(num_input_qubits: int, num_solutions: int = 1) -> int:
    N = 2 ** num_input_qubits
    theta = math.asin(math.sqrt(num_solutions / N))
    return max(0, round(math.pi / (4 * theta) - 0.5))

def _phase_oracle(
    oracle: Oracle 
) -> QuantumCircuit:
    """
        put a minus sign on each answer
        out is the output qubit as a helper to flip the correct inputs
    """
    if len(oracle.output) != 1:
        raise ValueError("search needs a single yes/no output")
    
    out, mode = oracle.output[0]
    if mode == OutputMode.PHASE:
        return oracle.circuit
    
    qc = QuantumCircuit(oracle.circuit.num_qubits)

    """
        branch |00⟩|−⟩ : inputs ≠ 11 → no flip → +|00⟩|−⟩
        branch |01⟩|−⟩ : inputs ≠ 11 → no flip → +|01⟩|−⟩
        branch |10⟩|−⟩ : inputs ≠ 11 → no flip → +|10⟩|−⟩
        branch |11⟩|−⟩ : inputs = 11 → flip    → −|11⟩|−⟩
    """

    qc.x(out)
    qc.h(out)
    qc.compose(oracle.circuit, inplace=True)
    qc.h(out)
    qc.x(out)

    return qc

def _diffuser(
    num_qubits: int, 
    inputs: list[int]
) -> QuantumCircuit:
    """
    turn minus signs into high probability
    flip across the average
    algorithm negates the total
    """
    qc = QuantumCircuit(num_qubits)

    qc.h(inputs)
    qc.x(inputs)

    qc.h(inputs[-1])
    qc.mcx(inputs[:-1], inputs[-1])
    qc.h(inputs[-1])

    qc.x(inputs)
    qc.h(inputs)

    return qc
    
def _grover_circuit(
    oracle: Oracle, 
    k: int
) -> tuple[QuantumCircuit, list[int]]:
    """
    contruct the circuit by looping k times and glue phase oracle and diffuser gates
    to the circuit each loop
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    """
    inputs: apply gates and measurements to all input qubits.
    oracle.inputs: identify which measured bits belong to x and which belong to y.
    """
    inputs = oracle.inputs

    inputs = [
        qubits
        for var_values in inputs.values()
        for qubits in var_values
    ]

    if not inputs:
        raise ValueError("need at least one qubit")

    qc = QuantumCircuit(oracle.circuit.num_qubits)

    # initialize with equal probability
    qc.h(inputs)

    phase_oracle = _phase_oracle(oracle)
    diffuser = _diffuser(qc.num_qubits, inputs)

    for _ in range(k):
        qc.compose(phase_oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    return qc, inputs

def search(
    params: dict[str, Iterable[int]],
    pred: Bit # expression tree for the condition
) -> dict[str, int] | None:
    """
    search for solutions in the space of params that satisfy pred

    the search domain is the cartesian product of each parameter's search space specified in params.

    return one solution set that satisfy the predicate
    """

    allowed = {
        name: set(values)
        for name, values in params.items()
    }

    oracle = compile_oracle(pred)

    k = k = _optimal_k(sum(len(q) for q in oracle.inputs.values()))

    # inputs: the sum of the variables' binary width
    circuit, inputs = _grover_circuit(oracle, k=k)

    # add classical bits to the circuit to store the results
    measured = ClassicalRegister(len(inputs), "measured")
    circuit.add_register(measured)

    for index, qubit in enumerate(inputs):
        circuit.measure(qubit, measured[index])

    #run the quantum simulation on a classical computer's CPU
    simulator = AerSimulator()
    execuble = transpile(circuit, simulator)
    result = simulator.run(execuble, shots=1024).result()
    counts = result.get_counts()

    assignments = []

    # most frequent outcomes first: after amplification, solutions dominate
    for bitstring, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        bits = bitstring[::-1]

        measured_by_qubit = {}

        for position, qubit in enumerate(inputs):
            measured_by_qubit[qubit] = int(bits[position])

        assignment = {}

        for name, qubits in oracle.inputs.items():
            value = 0

            # decode: turn bits in decimals
            for bit_index, qubit in enumerate(qubits):
                bit = measured_by_qubit[qubit]
                value += bit * (2 ** bit_index)

            assignment[name] = value

        valid = True

        for name, value in assignment.items():
            if value not in allowed[name]:
                valid = False
                break

        if not valid:
            continue

        if evaluate(pred, assignment):
            return assignment

    return None

# 3x+y=5
# (1,2)