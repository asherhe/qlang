from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

from qiskit.circuit import QuantumCircuit, Qubit

@dataclass(frozen=True, slots=True)
class Q:
    """a single qubit, with extra metadata"""
    qubit: Qubit
    # this qubit is only used by one operation. metadata only: excluded from equality so that
    # the same physical qubit still matches its pool entry regardless of the annotation
    one_use: bool = field(default=False, compare=False)

@dataclass
class QCtx:
    """packages the construction environment of a quantum circuit"""
    circuit: QuantumCircuit
    pool: AncillaPool

    def pick_one_use(self, *qubits: Q) -> Q:
        """pick a one-use qubit from the arguments. allocate a new qubit in the pool if none are one-use"""
        for q in qubits:
            if q.one_use:
                return q
        return self.pool.alloc()

    # gate wrappers
    def x(self, q: Q): self.circuit.x(q.qubit)
    def cx(self, a: Q, t: Q): self.circuit.cx(a.qubit, t.qubit)
    def ccx(self, a: Q, b: Q, t: Q): self.circuit.ccx(a.qubit, b.qubit, t.qubit)
    def mcx(self, controls: list[Q], t: Q): self.circuit.mcx([c.qubit for c in controls], t.qubit)

class AncillaPool:
    """
    manages allocation of ancilla qubits.

    ancilla qubits are recycled across the circuit to minimize the number of qubits required
    """

    circuit: QuantumCircuit

    # all ancillas ever created
    ancillas: list[Q]

    free_qubits: list[Q]  # currently available ancillas
    used_qubits: list[Q]  # ancillas currently in use

    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.ancillas = []
        self.free_qubits = []
        self.used_qubits = []

    def alloc(self) -> Q:
        """give an ancilla qubit that is set to |0>"""
        if self.free_qubits:
            q = self.free_qubits.pop()
        else:
            raw_q = Qubit()
            self.circuit.add_bits([raw_q])
            q = Q(qubit=raw_q, one_use=True)
            self.ancillas.append(q)
        self.used_qubits.append(q)
        return q

    def allocs(self, count: int) -> list[Q]:
        """allocate a many qubits"""
        return [self.alloc() for i in range(count)]

    def free(self, q: Q):
        """return an ancilla qubit ot the pool"""
        if q not in self.used_qubits:
            raise ValueError("trying to free an untracked or already freed qubit")
        self.used_qubits.remove(q)
        self.free_qubits.append(q)

    def mark(self) -> int:
        """returns the current allocation marker index"""
        return len(self.used_qubits)

    def release_since(self, mark: int):
        """releases all ancillas allocated since the given mark back to the free list"""
        while len(self.used_qubits) > mark:
            q = self.used_qubits.pop()
            if q not in self.free_qubits:
                self.free_qubits.append(q)

    @contextmanager
    def scratch(self, scratch_count: int = 1, output_count: int = 1):
        """
        context manager for temporary workspace with automatic gate uncomputation and recycling

        all work in the scratch context is uncomputed on exit, including changes to input qubits.
        only output ancillas are preserved and handed to the caller
        """
        circuit_mark = len(self.circuit.data)
        pool_mark = self.mark()

        scratch_qs = [self.alloc() for i in range(scratch_count)]
        output_qs = [self.alloc() for i in range(output_count)]
        
        yield_target = (tuple(scratch_qs), tuple(output_qs))

        compute_end_idx = None

        def harvest():
            """
            call this inside the block when the compute phase is complete

            after this, you should only copy results to output ancillas
            """
            nonlocal compute_end_idx
            compute_end_idx = len(self.circuit.data)

        try:
            yield yield_target
        finally:
            # fallback to current poitn if harvest() not called
            end_idx = compute_end_idx if compute_end_idx is not None else len(self.circuit.data)
            # get new computations in context block
            scratch_circuit = self.circuit.data[circuit_mark : end_idx]
            output_set = {q.qubit for q in output_qs}
            for inst in reversed(scratch_circuit):
                # this computation has nothing to do with output bits
                # TODO: we are blindly trusting that the user is not doing anything sneaky wtih the
                # output bits, like mutating the workplace state with it. add guards in the future?
                if len(set(inst.qubits) & output_set) == 0:
                    self.circuit.append(inst.operation.inverse(), inst.qubits, inst.clbits)

            for q in scratch_qs:
                self.free(q)

