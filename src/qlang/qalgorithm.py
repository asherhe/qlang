from .expression import ASTNode, Bit, UInt

from collections.abc import Iterable
from typing import cast

from .compile import Oracle, compile_oracle, OutputMode

from .evaluate import evaluate

from .optimize import optimize_expr

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister, transpile

import math

# how many of the most frequent measurements to check for each k
# small on purpose: only checking the top results means amplification has to
# find the answer. checking every result would just be a classical brute force
# more than 1 because the oracle does not know about params, so an amplified
# solution outside the allowed values can rank first
TOP_RESULTS = 3

def _max_k(num_input_qubits: int) -> int:
    """
    the largest k worth trying

    the best k is about (pi / 4) * sqrt(N / M) for N inputs and M solutions.
    M = 1 gives the largest value, so no search should need much more than this
    the + 1 covers cases where the rounded value lands on a bad k: for a | b,
    N = 4 and M = 3, k = 1 has a 0% chance of a solution but k = 2 has 75%
    """
    N = 2 ** num_input_qubits
    return math.ceil(math.pi / 4 * math.sqrt(N)) + 1

def _phase_oracle(
    oracle: Oracle 
) -> QuantumCircuit:
    """
        put a minus sign on each answer
        out is the output qubit as a helper to flip the correct inputs
    """
    # compile_oracle marks solutions in phase itself and has no output qubit
    if not oracle.output:
        return oracle.circuit

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

def _try_k(
    oracle: Oracle,
    pred: Bit,
    allowed: dict[str, set[int]],
    k: int
) -> dict[str, int] | None:
    """
    run grover with k iterations and check the most frequent results

    return a valid solution, or None if none of the top results are solutions
    """
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

    # most frequent outcomes first: after amplification, solutions dominate
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    # stop at TOP_RESULTS: if the wrong k was used, the top results are not
    # solutions, and search moves on to the next k
    for bitstring, _ in ranked[:TOP_RESULTS]:
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

    # compile_oracle only accepts decomposed logic; evaluate below keeps using pred
    oracle = compile_oracle(cast(Bit, optimize_expr(pred)))

    num_input_qubits = sum(len(q) for q in oracle.inputs.values())

    # the best k depends on the number of solutions, which we don't know yet.
    # too many iterations overshoots and makes solutions unlikely, so start low
    # and stop at the first k whose top results include a solution
    # k starts at 1 because k = 0 is just random guessing
    for k in range(1, _max_k(num_input_qubits) + 1):
        assignment = _try_k(oracle, pred, allowed, k)

        if assignment is not None:
            return assignment

    # no k found a solution, so there most likely is none
    return None

# 3x+y=5
# (1,2)