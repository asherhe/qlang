"""QAOA cost encoding, circuit evaluation, and feasible sample selection."""

from importlib import import_module
from math import pi
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qlang.expression import const_bit, var_uint
from qlang.minimize import constraint_to_cost_op, decode_state, qaoa, qaoa_creator


@pytest.mark.parametrize(
    "state,widths,expected",
    [
        (0b000, {"x": 2, "p": 1}, {"x": 0, "p": 0}),
        (0b101, {"x": 2, "p": 1}, {"x": 1, "p": 1}),
        (0b101, {"p": 1, "x": 2}, {"p": 1, "x": 2}),
        (0b111, {"p": 1, "x": 2}, {"p": 1, "x": 3}),
    ],
)
def test_decode_state_uses_layout_order(state, widths, expected):
    assert decode_state(state, widths) == expected


def test_cost_operator_matches_two_bit_example(p, q):
    operator = constraint_to_cost_op(
        p ^ q, {"p": 1, "q": 1}, lambda assignment: assignment["p"], penalty=3.0
    )

    assert dict(operator.to_list()) == {"II": 2.0, "IZ": -0.5, "ZZ": 1.5}
    np.testing.assert_allclose(operator.to_matrix(), np.diag([3, 1, 0, 4]))


@pytest.mark.parametrize("penalty", [0.0, 0.5, 4.0])
def test_cost_operator_encodes_uint_equation_and_objective(penalty):
    x, y = var_uint("x", 2), var_uint("y", 2)
    operator = constraint_to_cost_op(
        x + y == 3, {"x": 2, "y": 2}, lambda assignment: assignment["x"] / 2, penalty
    )
    # x occupies the low two bits, so it varies fastest in basis order.
    expected = [
        x_value / 2 + penalty * (x_value + y_value != 3)
        for y_value in range(4)
        for x_value in range(4)
    ]

    assert operator.num_qubits == 4
    np.testing.assert_allclose(operator.to_matrix(), np.diag(expected), atol=1e-12)


@pytest.mark.parametrize("satisfied,energy", [(True, 0.0), (False, 2.5)])
def test_constant_constraint_retains_layout_width(satisfied, energy):
    operator = constraint_to_cost_op(const_bit(satisfied), {"x": 2}, penalty=2.5)

    assert operator.num_qubits == 2
    np.testing.assert_allclose(operator.to_matrix(), energy * np.eye(4))


def test_qaoa_zero_angles_score_uniform_distribution(p, q):
    operator = constraint_to_cost_op(p & q, {"p": 1, "q": 1})
    state_at, mean_cost = qaoa_creator(operator)

    np.testing.assert_allclose(state_at(0.0, 0.0).probabilities(), [0.25] * 4)
    assert mean_cost((0.0, 0.0)) == pytest.approx(0.75)


def test_qaoa_binds_angles_to_cost_and_mixer_in_order():
    operator = SparsePauliOp.from_list([("II", 2.0), ("IZ", -0.5), ("ZZ", 1.5)])
    state_at, mean_cost = qaoa_creator(operator)
    beta, gamma = 0.23, 0.61

    expected = QuantumCircuit(2)
    expected.h([0, 1])
    expected.rz(-gamma, 0)
    expected.rzz(3 * gamma, 0, 1)
    expected.rx(2 * beta, [0, 1])

    state = state_at(beta, gamma)
    assert state.equiv(Statevector.from_instruction(expected))
    assert mean_cost((beta, gamma)) == pytest.approx(
        np.dot(state.probabilities(), [3, 1, 0, 4])
    )
    # The same ansatz can still bind a different angle pair.
    assert mean_cost((0.0, 0.0)) == pytest.approx(2.0)


def test_qaoa_runs_real_circuit_and_decodes_mixed_inputs(p):
    x = var_uint("x", 2)
    constraint = p & (x == 2)

    assert qaoa([p, x], constraint, iterations=2) == {"p": 1, "x": 2}


@pytest.mark.xfail(
    strict=True, raises=KeyError, reason="a constant cost omits the gamma parameter"
)
def test_qaoa_accepts_constant_zero_cost(p):
    solution = qaoa([p], const_bit(True), iterations=2)

    assert solution in ({"p": 0}, {"p": 1})


@pytest.fixture
def sample_qaoa_counts(monkeypatch):
    """Make readout tests independent of simulator shot order and randomness."""
    module = import_module("qlang.minimize")

    def install(counts):
        sampled_at = []

        def create(cost_operator):
            def state_at(beta, gamma):
                def sample_counts(shots):
                    assert shots == 1024
                    sampled_at.append((beta, gamma))
                    return counts

                return SimpleNamespace(sample_counts=sample_counts)

            def mean_cost(angles):
                beta, gamma = angles
                return (beta - pi) ** 2 + gamma ** 2

            return state_at, mean_cost

        monkeypatch.setattr(module, "qaoa_creator", create)
        return sampled_at

    return install


def test_qaoa_feasibility_skips_invalid_samples(p, q, sample_qaoa_counts):
    sampled_at = sample_qaoa_counts({"00": 900, "01": 123, "10": 1})

    assert qaoa([p, q], p ^ q, iterations=2) == {"p": 1, "q": 0}
    assert sampled_at == [(pi, 0.0)]


def test_qaoa_picks_lowest_objective_among_feasible_samples(p, q, sample_qaoa_counts):
    sample_qaoa_counts({"00": 900, "01": 123, "10": 1})

    assert qaoa(
        [p, q], p ^ q, lambda assignment: assignment["p"], penalty=3.0, iterations=2
    ) == {"p": 0, "q": 1}


@pytest.mark.parametrize(
    "with_objective",
    [
        pytest.param(False, id="feasibility"),
        pytest.param(
            True,
            id="objective",
            marks=pytest.mark.xfail(
                strict=True,
                raises=ValueError,
                reason="objective selection calls min without a default for no feasible samples",
            ),
        ),
    ],
)
def test_qaoa_returns_none_when_no_feasible_sample(p, q, sample_qaoa_counts, with_objective):
    sample_qaoa_counts({"00": 1024})
    objective = (lambda assignment: assignment["p"]) if with_objective else None

    assert qaoa([p, q], p ^ q, objective, iterations=2) is None
