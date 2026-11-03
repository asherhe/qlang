"""Implemented oracle metadata and an explicit specification for the compiler stub."""

import pytest
from qiskit import QuantumCircuit

from qlang.compile import Oracle, OutputMode, compile_oracle


def test_oracle_metadata_and_ancilla_count():
    circuit = QuantumCircuit(5)
    oracle = Oracle(circuit, {"x": [0, 1]}, [(2, OutputMode.BIT_FLIP)], [3], [4])
    assert oracle.circuit is circuit
    assert oracle.inputs == {"x": [0, 1]}
    assert oracle.output == [(2, OutputMode.BIT_FLIP)]
    assert oracle.total_ancillas == 2


def test_ancilla_defaults_are_independent():
    first = Oracle(QuantumCircuit(1), {}, [])
    second = Oracle(QuantumCircuit(1), {}, [])
    first.clean_ancillas.append(0)
    first.dirty_ancillas.append(0)
    assert second.clean_ancillas == []
    assert second.dirty_ancillas == []
    assert second.total_ancillas == 0


def test_output_modes_are_distinct():
    assert OutputMode.BIT_FLIP is not OutputMode.PHASE


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="compile_oracle is an unimplemented stub returning None")
def test_compile_returns_oracle(p):
    assert isinstance(compile_oracle(p), Oracle)
