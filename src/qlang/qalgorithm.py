from .expression import ASTNode, Bit, UInt

from collections.abc import Iterable

from .compile import Oracle, compile_oracle, OutputMode

from qiskit import QuantumCircuit

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
    


def search(
    params: dict[str, Iterable[int]],
    pred: Bit,
) -> list[dict[str, int]]:
    """
    search for solutions in the space of params that satisfy pred

    the search domain is the cartesian product of each parameter's search space specified in params.

    return a list of all solution sets that satisfy the predicate
    """
    oracle = compile_oracle(pred)
    ...

