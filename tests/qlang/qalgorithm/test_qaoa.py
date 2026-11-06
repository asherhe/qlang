"""QAOA cost encoding, circuit evaluation, and feasible sample selection."""

from importlib import import_module
from math import pi
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qlang.expression import bits_uint, const_bit, const_uint, var_uint
from qlang.minimize import (
    constraint_to_cost_op,
    decode_state,
    evaluate_linear,
    expand_to_bits,
    extract_checker,
    extract_linear,
    linear_objective_terms,
    qaoa,
    qaoa_creator,
    squared_residuals,
    terms_to_op,
)


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
        p ^ q, {"p": 1, "q": 1}, bits_uint((p,)), penalty=3.0
    )

    assert dict(operator.to_list()) == {"II": 2.0, "IZ": -0.5, "ZZ": 1.5}
    np.testing.assert_allclose(operator.to_matrix(), np.diag([3, 1, 0, 4]))


@pytest.mark.parametrize("penalty", [0.0, 0.5, 4.0])
def test_cost_operator_encodes_uint_equation_and_objective(penalty):
    x, y = var_uint("x", 2), var_uint("y", 2)
    operator = constraint_to_cost_op(
        x + y == 3, {"x": 2, "y": 2}, 2 * x + y, penalty
    )
    # x occupies the low two bits, so it varies fastest in basis order.
    expected = [
        2 * x_value + y_value + penalty * (x_value + y_value != 3)
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


def test_qaoa_accepts_constant_zero_cost(p):
    solution = qaoa([p], const_bit(True), iterations=2)

    assert solution in ({"p": 0}, {"p": 1})


@pytest.fixture
def sample_qaoa_counts(monkeypatch):
    """Make readout tests independent of simulator shot order and randomness."""
    module = import_module("qlang.minimize")

    def install(counts):
        sampled_at = []
        operators = []

        def create(cost_operator):
            operators.append(cost_operator)
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
        return SimpleNamespace(sampled_at=sampled_at, operators=operators)

    return install


def test_qaoa_feasibility_skips_invalid_samples(p, q, sample_qaoa_counts):
    captured = sample_qaoa_counts({"00": 900, "01": 123, "10": 1})

    assert qaoa([p, q], p ^ q, iterations=2) == {"p": 1, "q": 0}
    assert captured.sampled_at == [(pi, 0.0)]


def test_qaoa_picks_lowest_objective_among_feasible_samples(p, q, sample_qaoa_counts):
    sample_qaoa_counts({"00": 900, "01": 123, "10": 1})

    assert qaoa(
        [p, q], p ^ q, bits_uint((p,)), penalty=3.0, iterations=2
    ) == {"p": 0, "q": 1}


@pytest.mark.parametrize(
    "with_objective",
    [
        pytest.param(False, id="feasibility"),
        pytest.param(True, id="objective"),
    ],
)
def test_qaoa_returns_none_when_no_feasible_sample(p, q, sample_qaoa_counts, with_objective):
    sample_qaoa_counts({"00": 1024})
    objective = bits_uint((p,)) if with_objective else None

    assert qaoa([p, q], p ^ q, objective, iterations=2) is None


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda x, y: const_uint(7), (7, {})),
        (lambda x, y: x, (0, {"x": 1})),
        (lambda x, y: x + y + 2, (2, {"x": 1, "y": 1})),
        (lambda x, y: 3 * (x + 2) + y + x, (6, {"x": 4, "y": 1})),
        (lambda x, y: (x + 2) * 3, (6, {"x": 3})),
        (lambda x, y: 0 * x, (0, {})),
        (lambda x, y: x - y, NotImplemented),
    ],
)
def test_extract_linear_collects_constants_and_coefficients(x, y, build, expected):
    assert extract_linear(build(x, y)) == expected


def test_extract_linear_supports_bit_leaves(p):
    assert extract_linear(p) == (0, {"p": 1})
    assert extract_linear(const_bit(True)) == (True, {})
    assert extract_linear(~p) is NotImplemented


@pytest.mark.parametrize(
    "build",
    [
        lambda x, y: x * y,
        lambda x, y: (x + 1) * (y + 2),
        lambda x, y: x * y + 1,
    ],
)
def test_extract_linear_rejects_variable_products(x, y, build):
    assert extract_linear(build(x, y)) is NotImplemented


def test_extract_checker_propagates_unsupported_nested_expression(x, y):
    assert extract_checker((x - y) + 1) is NotImplemented
    assert extract_linear((x - y) + 1) is NotImplemented


def test_evaluate_linear_moves_both_sides_into_residual(x, y):
    constraint = 3 * x + y + 2 == x + 5

    assert evaluate_linear(constraint) == (-3, {"x": 2, "y": 1})
    # Extraction should be reusable and should not change the source expression.
    assert evaluate_linear(constraint) == (-3, {"x": 2, "y": 1})
    assert extract_linear(constraint.operands[0]) == (2, {"x": 3, "y": 1})


def test_evaluate_linear_cancels_repeated_variables(x):
    constant, coefficients = evaluate_linear(x + 2 == x + 5)

    assert constant == -3
    assert all(coefficient == 0 for coefficient in coefficients.values())


@pytest.mark.parametrize("build", [lambda x, y: x < y, lambda x, y: x - y == 1])
def test_evaluate_linear_returns_sentinel_for_unsupported_constraint(x, y, build):
    assert evaluate_linear(build(x, y)) is NotImplemented


@pytest.mark.parametrize(
    "coefficients,widths,expected",
    [
        ({"x": 3, "y": -1}, {"x": 2, "y": 1}, {0: 3, 1: 6, 2: -1}),
        ({"x": 3}, {"unused": 1, "x": 2}, {1: 3, 2: 6}),
        ({"x": 0}, {"x": 2}, {}),
    ],
)
def test_expand_to_bits_preserves_layout_and_signed_weights(coefficients, widths, expected):
    original_coefficients, original_widths = coefficients.copy(), widths.copy()

    assert expand_to_bits(coefficients, widths) == expected
    assert coefficients == original_coefficients
    assert widths == original_widths


def test_expand_to_bits_rejects_variable_missing_from_layout():
    with pytest.raises(ValueError, match="missing"):
        expand_to_bits({"missing": 1}, {"x": 2})


@pytest.mark.parametrize("penalty", [0.0, 1.0, 4.0])
def test_squared_residual_scales_entire_identity_coefficient(penalty):
    operator = terms_to_op(squared_residuals(-1.0, {0: 2.0}, penalty), 1)

    # (2*b - 1)^2 is 1 for both bit values.
    np.testing.assert_allclose(operator.to_matrix(), penalty * np.eye(2), atol=1e-12)


def test_squared_residual_counts_each_pair_once():
    operator = terms_to_op(squared_residuals(-1.0, {0: 1.0, 1: 1.0}, 1.0), 2)

    # (a + b - 1)^2 has energies [1, 0, 0, 1].
    assert dict(operator.to_list()) == {"II": 0.5, "ZZ": 0.5}
    np.testing.assert_allclose(operator.to_matrix(), np.diag([1, 0, 0, 1]))


def test_squared_residual_handles_constant_without_weights():
    assert dict(squared_residuals(-3.0, {}, 2.0)) == {0: 18.0}


def test_linear_objective_terms_reproduce_signed_linear_cost():
    terms = linear_objective_terms(4.0, {0: 2.0, 2: -3.0})
    operator = terms_to_op(terms, 3)
    expected = [
        4 + 2 * (index & 1) - 3 * ((index >> 2) & 1)
        for index in range(8)
    ]

    np.testing.assert_allclose(operator.to_matrix(), np.diag(expected), atol=1e-12)


@pytest.mark.parametrize("terms", [{}, {0: 0.0, 1: 0.0}])
def test_terms_to_op_preserves_width_for_zero_operator(terms):
    operator = terms_to_op(terms, 3)

    assert operator.num_qubits == 3
    np.testing.assert_allclose(operator.to_matrix(), np.zeros((8, 8)))


def test_terms_to_op_uses_qiskit_label_order():
    operator = terms_to_op({0: 2.0, 1: -0.5, 5: 1.5}, 3)

    assert dict(operator.to_list()) == {"III": 2.0, "IIZ": -0.5, "ZIZ": 1.5}


@pytest.mark.parametrize("with_objective", [False, True])
def test_qaoa_linear_path_passes_squared_cost_to_ansatz(
    monkeypatch, sample_qaoa_counts, with_objective
):
    x, y = var_uint("x", 2), var_uint("y", 2)
    captured = sample_qaoa_counts({"0011": 1023, "1100": 1})
    module = import_module("qlang.minimize")

    def unexpected_fallback(*args, **kwargs):
        pytest.fail("A linear equality should use direct coefficient compilation")

    monkeypatch.setattr(module, "constraint_to_cost_op", unexpected_fallback)
    objective = 2 * x + y if with_objective else None
    solution = qaoa([x, y], x + y == 3, objective, penalty=4.0, iterations=2)

    expected = [
        (2 * x_value + y_value if with_objective else 0)
        + 4 * (x_value + y_value - 3) ** 2
        for y_value in range(4)
        for x_value in range(4)
    ]
    np.testing.assert_allclose(
        captured.operators[0].to_matrix(), np.diag(expected), atol=1e-12
    )
    assert solution == ({"x": 0, "y": 3} if with_objective else {"x": 3, "y": 0})


def test_qaoa_rejects_nonlinear_objective_on_linear_path(sample_qaoa_counts):
    x, y = var_uint("x", 2), var_uint("y", 2)
    sample_qaoa_counts({"0011": 1024})

    with pytest.raises(ValueError, match="linear objective"):
        qaoa([x, y], x + y == 3, objective=x * y, iterations=2)


def test_qaoa_fallback_preserves_nonlinear_objective(p, q, sample_qaoa_counts):
    objective = bits_uint((p,)) * bits_uint((q,))
    captured = sample_qaoa_counts({"11": 1000, "01": 24})

    solution = qaoa([p, q], p | q, objective, penalty=3.0, iterations=2)

    np.testing.assert_allclose(
        captured.operators[0].to_matrix(), np.diag([3, 0, 0, 1]), atol=1e-12
    )
    assert solution == {"p": 1, "q": 0}
