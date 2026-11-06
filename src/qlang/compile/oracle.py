from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from qiskit import QuantumCircuit

class OutputMode(Enum):
    BIT_FLIP = auto()
    PHASE = auto()

@dataclass
class Oracle:
    """container class for compiled oracle circuit outputs"""

    # compiled circuit result
    circuit: QuantumCircuit
    
    # variable name -> qubit indices
    # orphan Bits (not in a UInt) are treated as 1-bit width UInts
    inputs: dict[str, list[int]] 

    # External outputs; empty for compile_oracle's phase-only interface.
    output: list[tuple[int, OutputMode]]

    # track ancilla qubits
    clean_ancillas: list[int] = field(default_factory=list)
    dirty_ancillas: list[int] = field(default_factory=list)

    # (symbolic operation name, first gate index, gate index after the operation)
    # in the forward computation of circuit. Consumers may use this only for
    # visualization; circuit remains the executable phase oracle.
    operation_spans: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def total_ancillas(self) -> int:
        return len(self.clean_ancillas) + len(self.dirty_ancillas)

