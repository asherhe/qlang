"""Oracle container metadata and compile_oracle return-type tests."""

import pytest

from qiskit import QuantumCircuit

from qlang.compile import CompiledCircuit, Oracle, OutputMode, compile_circuit, compile_oracle
from qlang.expression import const_bit, var_bit, var_uint


def test_oracle_metadata_and_ancilla_count():
    circuit = QuantumCircuit(5)
    oracle = Oracle(circuit, {"x": [0, 1]}, [(2, OutputMode.BIT_FLIP)], [3], [4])
    assert oracle.circuit is circuit
    assert oracle.inputs == {"x": [0, 1]}
    assert oracle.output == [(2, OutputMode.BIT_FLIP)]
    assert oracle.total_ancillas == 2


def test_ancilla_defaults_are_independent():
    first, second = Oracle(QuantumCircuit(1), {}, []), Oracle(QuantumCircuit(1), {}, [])
    first.clean_ancillas.append(0)
    first.dirty_ancillas.append(0)
    assert second.clean_ancillas == []
    assert second.dirty_ancillas == []
    assert second.total_ancillas == 0


def test_output_modes_are_distinct():
    assert OutputMode.BIT_FLIP is not OutputMode.PHASE


def test_compile_returns_oracle():
    assert isinstance(compile_oracle(var_bit("p")), Oracle)


def test_compile_circuit_returns_forward_layout():
    compiled = compile_circuit(var_bit("p"))

    assert isinstance(compiled, CompiledCircuit)
    assert compiled.inputs == {"p": (0,)}
    assert compiled.outputs == (0,)
    assert compiled.clean_ancillas == ()
    assert compiled.dirty_ancillas == ()
    assert compiled.circuit.num_qubits == 1


def test_compile_circuit_returns_all_uint_output_bits():
    compiled = compile_circuit(var_uint("x", 3))

    assert compiled.inputs == {"x": (0, 1, 2)}
    assert compiled.outputs == (0, 1, 2)


def test_compile_oracle_rejects_uint_expression():
    with pytest.raises(TypeError, match="compile_oracle expects a Bit expression"):
        compile_oracle(var_uint("x", 3))


def test_compile_allocates_only_constants_the_expression_uses():
    no_const = compile_oracle(var_bit("p"))
    false_const = compile_oracle(const_bit(False))
    true_const = compile_oracle(const_bit(True))

    assert no_const.total_ancillas == 0
    assert no_const.circuit.num_qubits == 1
    assert false_const.total_ancillas == 1
    assert false_const.circuit.num_qubits == 1
    assert true_const.total_ancillas == 1
    assert true_const.circuit.num_qubits == 1
