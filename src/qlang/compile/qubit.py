from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Iterable

from qiskit.circuit import QuantumCircuit, Qubit

@dataclass(frozen=True, slots=True)
class Q:
    """a single qubit, with extra metadata"""
    qubit: Qubit
    # this qubit is only used by one operation. metadata only: excluded from equality so that
    # the same physical qubit still matches its pool entry regardless of the annotation
    one_use: bool = field(default=False, compare=False)

    # whether this qubit represents a known constant at compile-time
    is_const: bool = False
    # constant value (0 or 1)
    const_value: int | None = None

class QCtx:
    """packages the construction environment of a quantum circuit"""
    circuit: QuantumCircuit
    pool: AncillaPool

    const_zero: Q
    const_one: Q

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit
        self.pool = AncillaPool(circuit)

        self.const_zero = replace(self.pool.alloc(), is_const=True, const_value=0)
        self.const_one = replace(self.pool.alloc(), is_const=True, const_value=1)
        self.x(self.const_one)

    def pick_one_use(self, *qubits: Q) -> Q:
        """pick a one-use qubit from the arguments. allocate a new qubit in the pool if none are one-use"""
        for q in qubits:
            if q.one_use:
                return q
        return self.pool.alloc()

    def make_mutable_consts(self, a: list[Q]) -> list[Q]:
        """replace constant qubits with newly allocated, mutable qubits with the same value"""
        qout = []
        for q in a:
            if q.is_const:
                qnew = self.pool.alloc()
                qout.append(qnew)
                if q.const_value:
                    self.x(qnew)
            else:
                qout.append(q)
        return qout

    def free_mutable_consts(self, a: list[Q]):
        """deallocate any mutable consts allocated by make_mutable_consts"""
        for q in a:
            if q.is_const and q not in (self.const_zero, self.const_one):
                if q.const_value:
                    self.x(q)
                self.pool.free(q)

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
        self._allocation_serial = 0
        self._allocation_ids: dict[int, int] = {}

    def alloc(self) -> Q:
        """give an ancilla qubit that is set to |0>"""
        if self.free_qubits:
            stored = self.free_qubits.pop()
            q = Q(stored.qubit)
        else:
            raw_q = Qubit()
            self.circuit.add_bits([raw_q])
            q = Q(qubit=raw_q)
            self.ancillas.append(q)
        self.used_qubits.append(q)
        self._allocation_serial += 1 
        self._allocation_ids[id(q)] = self._allocation_serial
        return q

    def allocs(self, count: int) -> list[Q]:
        """allocate a many qubits"""
        return [self.alloc() for i in range(count)]

    def free(self, q: Q):
        """reclaim a clean, currently allocated ancilla qubit to the pool"""
        for i, used in enumerate(self.used_qubits):
            if used == q:
                self.used_qubits.pop(i)
                # reset all other metadata on the qubit
                self.free_qubits.append(Q(q.qubit))
                return
        raise ValueError("trying to free an untracked or already freed qubit")

    def mark(self) -> int:
        """returns the current allocation marker index"""
        return self._allocation_serial

    def release_since(self, mark: int, keep: Iterable[Q] = ()):
        """releases all ancillas allocated since the given mark back to the free list"""
        keep_set = {q.qubit for q in keep}
        for q in list(self.used_qubits):
            if self._allocation_ids[id(q)] > mark and q.qubit not in keep_set:
                self.free(q)

    def _uncompute_since(self, circuit_mark: int, protected: set[Qubit]) -> None:
        """append inverses of new gates since a given point in the circuit"""
        instructions = list(self.circuit.data[circuit_mark:])
        for inst in reversed(instructions):
            if set(inst.qubits) & protected:
                continue
            self.circuit.append(inst.operation.inverse(), inst.qubits, inst.clbits)
    
    @staticmethod
    def _validate_outputs_are_not_controls(
        instructions, output_set: set[Qubit]
    ) -> None:
        """outputs may be targets of copied results, but never controls"""
        for inst in instructions:
            touched = set(inst.qubits) & output_set
            if not touched:
                continue
            ncontrols = getattr(inst.operation, "num_ctrl_qubits", 0)
            controls = set(inst.qubits[:ncontrols])
            if controls & output_set:
                raise ValueError(
                    "scratch outputs cannot control operations inside the scratch block"
                )
            # controlled gates: the final wire is the target
            # uncontrolled gates: only a one-qubit operation is unambiguous
            if ncontrols:
                if touched != {inst.qubits[-1]}:
                    raise ValueError(
                        "scratch outputs may only be targets inside the scratch block"
                    )
            elif len(inst.qubits) != 1:
                raise ValueError(
                    "scratch outputs cannot participate in multi-qubit uncontrolled gates"
                )

    @contextmanager
    def scratch(self, scratch_count: int = 1, output_count: int = 1):
        """
        context manager for temporary workspace with automatic gate uncomputation and recycling

        all work in the scratch context is uncomputed on exit, including changes to non-ancilla
        qubits in the workspace. only output ancillas are preserved and handed to the caller.

        do NOT use output qubits as controls here, this will break the uncompute operations

        preserve is for input qubits deliberately converted into intermediate result bits
        """
        circuit_mark = len(self.circuit.data)
        pool_mark = self.mark()
        scratch_qs = self.allocs(scratch_count)
        output_qs = self.allocs(output_count)
        output_set = {q.qubit for q in output_qs}

        try:
            yield scratch_qs, output_qs
        except BaseException:
            # roll back computations
            self._uncompute_since(circuit_mark, set())
            self.release_since(pool_mark)
            raise
        else:
            # get new computations in context block
            instructions = list(self.circuit.data[circuit_mark:])
            
            try:
                self._validate_outputs_are_not_controls(instructions, output_set)
            except BaseException:
                self._uncompute_since(circuit_mark, set())
                self.release_since(pool_mark)
                raise

            self._uncompute_since(circuit_mark, output_set)
            self.release_since(pool_mark, output_qs)

