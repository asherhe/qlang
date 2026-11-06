"""Regression coverage for search constants, complete assignments, and domains.

Simulation is replaced with fixed counts so these contract tests do not depend
on sampling luck. Expression validation, optimization, compilation, measurement
decoding, and final predicate evaluation remain real.
"""

from collections.abc import Mapping
from unittest.mock import Mock

import pytest
from qiskit import QuantumCircuit

import qlang.qalgorithm as algorithm
from qlang.compile import Oracle
from qlang.evaluate import evaluate
from qlang.expression import (
    Bit, BitType, UInt, UIntType, bits_uint, const_bit, const_uint,
    var_bit, var_uint,
)


@pytest.fixture
def no_quantum(monkeypatch):
    """Shortcuts and invalid arguments must not reach quantum work."""
    def forbidden(*args, **kwargs):
        pytest.fail("quantum work started before preprocessing finished")

    monkeypatch.setattr(algorithm, "compile_oracle", forbidden)
    monkeypatch.setattr(algorithm, "_grover_circuit", forbidden)
    monkeypatch.setattr(algorithm, "AerSimulator", forbidden)


@pytest.fixture
def no_grover(monkeypatch):
    """Constant expressions may compile, but must never enter Grover."""
    def forbidden(*args, **kwargs):
        pytest.fail("Grover started for a decided expression")

    monkeypatch.setattr(algorithm, "_grover_circuit", forbidden)
    monkeypatch.setattr(algorithm, "AerSimulator", forbidden)


@pytest.fixture
def measured_counts(monkeypatch):
    """Replace only the simulator/transpiler boundary with fixed measurements."""
    def configure(counts):
        simulator = Mock()
        simulator.run.return_value.result.return_value.get_counts.return_value = counts
        monkeypatch.setattr(algorithm, "AerSimulator", lambda **kwargs: simulator)
        monkeypatch.setattr(algorithm, "transpile", lambda circuit, backend: circuit)

    return configure


@pytest.mark.parametrize(
    "value, expected", [(False, None), (True, {})], ids=["false", "true"]
)
def test_constant_without_parameters(value, expected, no_quantum):
    assert algorithm.search({}, const_bit(value)) == expected


@pytest.mark.parametrize("value", [False, True])
def test_constant_with_extra_parameters(value, no_quantum):
    result = algorithm.search(
        {"extra": [7, 4], "other": [9]}, const_bit(value)
    )
    assert result == ({"extra": 4, "other": 9} if value else None)


@pytest.mark.parametrize("value", [0, 3])
def test_uint_constant_uses_nonzero_truth(value, no_grover):
    assert algorithm.search({}, const_uint(value)) == ({} if value else None)


@pytest.mark.parametrize("tautology", [False, True])
def test_simplified_constant_keeps_original_variables(tautology, no_quantum, p):
    pred = p | ~p if tautology else p ^ p
    result = algorithm.search({"p": [1]}, pred)
    assert result == ({"p": 1} if tautology else None)
    if result is not None:
        assert evaluate(pred, result)


def test_constant_uint_comparison_does_not_enter_grover(no_grover):
    assert algorithm.search({}, (const_uint(2) + 1) == 3) == {}


@pytest.mark.parametrize("value", [False, True])
def test_empty_extra_domain_makes_constant_search_empty(value, no_quantum):
    assert algorithm.search({"extra": []}, const_bit(value)) is None


@pytest.mark.parametrize("empty_name", ["p", "extra"])
def test_any_empty_domain_prevents_compilation(empty_name, no_quantum, p):
    params = {"p": [0, 1], "extra": [4]}
    params[empty_name] = []
    assert algorithm.search(params, p) is None


def test_constant_shortcut_detects_optimizer_disagreement(monkeypatch, no_quantum):
    monkeypatch.setattr(algorithm, "optimize_expr", lambda expr: const_bit(False))
    with pytest.raises(RuntimeError, match="disagrees"):
        algorithm.search({}, const_bit(True))


@pytest.mark.parametrize("value", [0, 1])
def test_inputless_compiler_result_does_not_enter_grover(
    value, monkeypatch, no_quantum, p
):
    # Exercise the defensive branch even when the initial optimizer keeps p.
    oracle = Oracle(circuit=QuantumCircuit(), inputs={}, output=[])
    monkeypatch.setattr(algorithm, "compile_oracle", lambda expr: oracle)
    assert algorithm.search({"p": [value]}, p) == (
        {"p": 1} if value else None
    )


@pytest.mark.parametrize("default_p", [0, 1])
def test_search_restores_eliminated_variables(
    default_p, p, q, measured_counts
):
    pred = (p ^ p) | q
    measured_counts({"1": 1024})

    result = algorithm.search(
        {"p": [default_p], "q": [0, 1], "extra": [9, 4]}, pred
    )

    assert result == {"p": default_p, "q": 1, "extra": 4}
    assert evaluate(pred, result)


def test_search_restores_eliminated_uint(measured_counts):
    x = var_uint("x", 3)
    q = var_bit("q")
    pred = (x.bits[0] ^ x.bits[0]) | q
    measured_counts({"1": 1024})

    result = algorithm.search({"x": [7], "q": [0, 1]}, pred)

    assert result == {"x": 7, "q": 1}
    assert evaluate(pred, result)


def test_readout_completes_each_candidate_without_mutating_defaults(
    p, q, measured_counts
):
    # The first measurement is false; the second must get a fresh assignment.
    pred = (p ^ p) | q
    oracle = Oracle(circuit=QuantumCircuit(1), inputs={"q": [0]}, output=[])
    defaults = {"p": 1, "q": 0, "extra": 8}
    allowed = {"p": {1}, "q": {0, 1}, "extra": {8}}
    measured_counts({"0": 900, "1": 124})

    assert algorithm._try_k(oracle, pred, allowed, defaults, k=0) == {
        "p": 1, "q": 1, "extra": 8
    }
    assert defaults == {"p": 1, "q": 0, "extra": 8}


def test_readout_filters_disallowed_measured_values(measured_counts):
    x = var_uint("x", 2)
    oracle = Oracle(
        circuit=QuantumCircuit(2), inputs={"x": [0, 1]}, output=[]
    )
    # LSB first decoding: "10" is 2, "01" is 1.
    measured_counts({"10": 900, "01": 124})

    assert algorithm._try_k(
        oracle, x != 0, {"x": {1}}, {"x": 1}, k=0
    ) == {"x": 1}


def test_readout_returns_none_when_no_observed_assignment_is_valid(
    p, measured_counts
):
    oracle = Oracle(circuit=QuantumCircuit(1), inputs={"p": [0]}, output=[])
    measured_counts({"0": 1024})
    assert algorithm._try_k(
        oracle, p, {"p": {0, 1}}, {"p": 1}, k=0
    ) is None


def test_evaluator_still_rejects_incomplete_original_assignment(p, q):
    with pytest.raises(ValueError, match="not assigned"):
        evaluate((p ^ p) | q, {"q": 1})


@pytest.mark.parametrize("simplified", [False, True])
def test_missing_domain_is_rejected_before_optimization(
    simplified, no_quantum, p
):
    pred = p | ~p if simplified else p
    with pytest.raises(ValueError, match="missing.*p"):
        algorithm.search({}, pred)


def test_missing_domain_inside_bits_wrapper(no_quantum):
    x = var_uint("x", 2)
    pred = bits_uint((x.bits[1],)) != 0
    with pytest.raises(ValueError, match="missing.*x"):
        algorithm.search({}, pred)


def test_unrelated_global_variable_does_not_require_domain(no_quantum):
    var_uint("unrelated", 8)
    p = var_bit("p")
    assert algorithm.search({"p": [1]}, p | ~p) == {"p": 1}


def test_shared_variables_and_wrapper_parent_schema():
    x = var_uint("x", 3)
    p = var_bit("p")
    shared = p & x.bits[2]
    pred = bits_uint((shared, shared)) != 0
    assert algorithm._validate_widths(pred) == {"p": 1, "x": 3}


def test_distinct_declarations_with_same_name_are_rejected(no_quantum):
    x = UInt(uint_type=UIntType.VAR, name="x", width=2)
    # Simulate a malformed conflicting declaration without changing the registry.
    other = UInt(uint_type=UIntType.VAR, name="other", width=2)
    other.name = "x"
    with pytest.raises(ValueError, match="multiple variable declarations"):
        algorithm.search({"x": [0]}, x == other)


@pytest.mark.parametrize("params", [None, [], [("p", [0, 1])], 42])
def test_params_must_be_mapping(params, no_quantum):
    with pytest.raises(TypeError, match="mapping"):
        algorithm.search(params, const_bit(True))


@pytest.mark.parametrize("name", ["", "  ", 3, None])
def test_parameter_names_must_be_nonempty_strings(name, no_quantum):
    with pytest.raises(ValueError, match="names"):
        algorithm.search({name: [0]}, const_bit(True))


@pytest.mark.parametrize("domain", [None, 42, 1.5])
def test_domain_must_be_iterable(domain, no_quantum, p):
    with pytest.raises(TypeError, match="iterable"):
        algorithm.search({"p": domain}, p)


@pytest.mark.parametrize("value", [True, False, 1.5, "1", None, [], {}])
@pytest.mark.parametrize("extra", [False, True], ids=["referenced", "extra"])
def test_domain_values_must_be_actual_integers(value, extra, no_quantum, p):
    params = {"p": [0, 1]}
    params["extra" if extra else "p"] = [value]
    with pytest.raises(TypeError, match="integers"):
        algorithm.search(params, p)


@pytest.mark.parametrize("extra", [False, True], ids=["referenced", "extra"])
def test_negative_domain_values_are_rejected(extra, no_quantum, p):
    params = {"p": [0, 1]}
    params["extra" if extra else "p"] = [-1]
    with pytest.raises(ValueError, match="negative"):
        algorithm.search(params, p)


@pytest.mark.parametrize("width, value", [(1, 2), (3, 8), (3, 100)])
def test_domain_values_must_fit_referenced_variable(width, value, no_quantum):
    x = var_uint("x", width)
    with pytest.raises(ValueError, match="fit"):
        algorithm.search({"x": [value]}, x != 0)


def test_bit_domain_rejects_values_above_one(no_quantum, p):
    with pytest.raises(ValueError, match="fit"):
        algorithm.search({"p": [2]}, p)


@pytest.mark.parametrize("value", [0, 7])
def test_uint_domain_accepts_representable_boundaries(value, no_quantum):
    x = var_uint("x", 3)
    assert algorithm.search(
        {"x": [value]}, x.bits[0] | ~x.bits[0]
    ) == {"x": value}


def test_extra_parameter_has_no_inferred_bit_width(no_quantum):
    assert algorithm.search({"extra": [1024]}, const_bit(True)) == {
        "extra": 1024
    }


def test_invalid_domain_is_not_hidden_by_empty_domain(no_quantum, p):
    with pytest.raises(TypeError, match="integers"):
        algorithm.search({"p": [], "extra": ["bad"]}, p)


def test_missing_domain_is_not_hidden_by_empty_extra_domain(no_quantum, p):
    with pytest.raises(ValueError, match="missing.*p"):
        algorithm.search({"extra": []}, p)


def test_false_predicate_does_not_skip_domain_validation(no_quantum):
    with pytest.raises(TypeError, match="integers"):
        algorithm.search({"extra": [1.5]}, const_bit(False))


def test_generator_domains_are_consumed_once_and_input_mapping_is_unchanged(
    no_quantum, p
):
    class Once:
        def __init__(self, values):
            self.values = values
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("domain iterated more than once")
            return iter(self.values)

    domain = Once([1, 0, 1])
    extra = (value for value in [8, 5])
    params = {"p": domain, "extra": extra}

    assert algorithm.search(params, p | ~p) == {"p": 0, "extra": 5}
    assert domain.iterations == 1
    assert params["p"] is domain
    assert params["extra"] is extra
    assert set(params) == {"p", "extra"}


def test_duplicate_domain_values_do_not_change_results(no_quantum, p):
    assert algorithm.search({"p": [1, 1]}, p | ~p) == {"p": 1}


def test_readonly_mapping_is_accepted(no_quantum, p):
    from types import MappingProxyType

    params = MappingProxyType({"p": (1,)})
    assert isinstance(params, Mapping)
    assert algorithm.search(params, p | ~p) == {"p": 1}


@pytest.mark.parametrize("pred", [None, True, 1, "p", object()])
def test_invalid_predicate_types_are_rejected(pred, no_quantum):
    with pytest.raises(TypeError):
        algorithm.search({}, pred)


def test_malformed_original_expression_is_rejected_before_optimization(
    monkeypatch, no_quantum
):
    malformed = Bit(bit_type=BitType.NOT)
    pred = malformed & const_bit(False)

    def forbidden(expr):
        pytest.fail("optimization ran before expression validation")

    monkeypatch.setattr(algorithm, "optimize_expr", forbidden)
    with pytest.raises(ValueError, match="operand"):
        algorithm.search({}, pred)

