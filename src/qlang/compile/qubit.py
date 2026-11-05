from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit, Qubit

@dataclass(frozen=True, slots=True)
class Q:
    """a single qubit, with extra metadata"""
    qubit: Qubit
    # this qubit is only used by one operation
    one_use: bool = False

@dataclass
class QCtx:
    """packages the construction environment of a quantum circuit"""
    circuit: QuantumCircuit
    pool: AncillaPool

    # gate wrappers
    def x(self, q: Q): self.circuit.x(q.qubit)
    def cx(self, a: Q, t: Q): self.circuit.cx(a.qubit, t.qubit)
    def ccx(self, a: Q, b: Q, t: Q): self.circuit.ccx(a.qubit, b.qubit, t.qubit)

class AncillaPool:
    """
    manages allocation of ancilla qubits.

    ancilla qubits are recycled across the circuit to minimize the number of qubits required
    """
    ...

