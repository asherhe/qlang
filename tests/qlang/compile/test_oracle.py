"""Oracle container metadata and compile_oracle return-type tests."""

from qiskit import QuantumCircuit

from qlang.compile import Oracle, OutputMode, compile_oracle
from qlang.expression import const_bit, var_bit


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
