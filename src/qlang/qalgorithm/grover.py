from collections.abc import Iterable, Mapping
from typing import cast

from ..compile import Oracle, compile_oracle, OutputMode, validate_expression
from ..expression import ASTNode, Bit, UInt, BitType, UIntType, const_bit
from ..evaluate import evaluate
from ..optimize import optimize_expr

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister, transpile

import math
import random

# BBHT: growth rate of the limit on k
GROWTH = 6 / 5

# BBHT: total iterations allowed, as a multiple of sqrt(N)
BUDGET = 9

def _validate_widths(expr: Bit) -> dict[str, int]: 
    """collect original variables from a validated graph, alongside their widths"""
    widths: dict[str, int] = {}
    declarations: dict[str, ASTNode] = {}
    visited: set[int] = set()
    pending: list[ASTNode] = [expr]

    while pending: 
        node = pending.pop()
        if id(node) in visited: 
            continue
        visited.add(id(node))

        width = None
        if isinstance(node, Bit) and node.bit_type == BitType.VAR: 
            width = 1
        elif isinstance(node, UInt) and node.uint_type == UIntType.VAR: 
            width = node.width

        if width is not None: 
            name = cast(str, node.name)
            prev = declarations.get(name)
            if prev is not None and prev is not node: 
                raise ValueError(f"multiple variable declarations use name {name!r}")
            declarations[name] = node
            widths[name] = width
        pending.extend(node.children)
    return widths

def _prepare_domains(
    params: Mapping[str, Iterable[int]],
    widths: dict[str, int],
): 
    """validate parameter domains"""
    for name in params: 
        if not isinstance(name, str) or not name.strip(): 
            raise ValueError("parameter names must be nonempty strings")
    missing = widths.keys() - params.keys()
    if missing: 
        raise ValueError(f"missing parameter domains: {', '.join(sorted(missing))}")

    allowed: dict[str, set[int]] = {}
    for name, values in params.items(): 
        try: 
            vals = tuple(values)
        except TypeError as exc: 
            raise TypeError(f"domain for {name!r} must be iterable") from exc
        width = widths.get(name)
        for value in vals: 
            if type(value) is not int: 
                raise TypeError(f"domain for {name!r} must contain integers")
            if value < 0: 
                raise ValueError(f"domain for {name!r} cannot contain negative values")
            if width is not None and value >= (1 << width): 
                raise ValueError(f"value {value} does not fit {name!r}'s {width}-bit width")
        allowed[name] = set(vals)
    return allowed

def _restrict_domains(
    pred: Bit, 
    allowed: dict[str, set[int]],
) -> Bit: 
    """
    given a predicate and a prepared domain, generate an effective predicate where 
    the prepared domain is taken into consideration
    """
    variables: dict[str, Bit | UInt] = {}
    visited: set[int] = set()
    pending: list[ASTNode] = [pred]

    # traverse the AST to collect all variables
    while pending: 
        node = pending.pop()
        if id(node) in visited: 
            continue
        visited.add(id(node))

        if isinstance(node, Bit) and node.bit_type == BitType.VAR: 
            variables[cast(str, node.name)] = node
        elif isinstance(node, UInt) and node.uint_type == UIntType.VAR: 
            variables[cast(str, node.name)] = node
        pending.extend(node.children)

    effective = pred
    for name, variable in variables.items(): 
        values = allowed[name]
        width = variable.width if isinstance(variable, UInt) else 1
        
        # domain is already complete
        if len(values) == (1 << width): 
            continue
        if isinstance(variable, Bit): 
            inclusion = variable if 1 in values else ~variable
        else: 
            inclusion = const_bit(False)
            # creates a new condition to be inserted into the expression
            for value in values: 
                inclusion = inclusion | (variable == value)
        effective = effective & inclusion
    return effective

def _phase_oracle(
    oracle: Oracle 
) -> QuantumCircuit:
    """
        put a minus sign on each answer
        out is the output qubit as a helper to flip the correct inputs
    """
    # compile_oracle marks solutions in phase itself and has no output qubit
    if not oracle.output:
        return oracle.circuit
    if len(oracle.output) != 1:
        raise ValueError("search needs a single yes/no output")
    out, mode = oracle.output[0]
    if mode == OutputMode.PHASE:
        return oracle.circuit
    
    qc = QuantumCircuit(oracle.circuit.num_qubits)
    """
        branch |00⟩|−⟩ : inputs ≠ 11 → no flip → +|00⟩|−⟩
        branch |01⟩|−⟩ : inputs ≠ 11 → no flip → +|01⟩|−⟩
        branch |10⟩|−⟩ : inputs ≠ 11 → no flip → +|10⟩|−⟩
        branch |11⟩|−⟩ : inputs = 11 → flip    → −|11⟩|−⟩
    """
    qc.x(out)
    qc.h(out)
    qc.compose(oracle.circuit, inplace=True)
    qc.h(out)
    qc.x(out)

    return qc

def _diffuser(
    num_qubits: int, 
    inputs: list[int]
) -> QuantumCircuit:
    """
    turn minus signs into high probability
    flip across the average
    algorithm negates the total
    """
    qc = QuantumCircuit(num_qubits)

    qc.h(inputs)
    qc.x(inputs)

    qc.h(inputs[-1])
    qc.mcx(inputs[:-1], inputs[-1])
    qc.h(inputs[-1])

    qc.x(inputs)
    qc.h(inputs)

    return qc
    
def _grover_circuit(
    oracle: Oracle, 
    k: int
) -> tuple[QuantumCircuit, list[int]]:
    """
    contruct the circuit by looping k times and glue phase oracle and diffuser gates
    to the circuit each loop
    """
    if k < 0:
        raise ValueError("k must be non-negative")

    """
    inputs: apply gates and measurements to all input qubits.
    oracle.inputs: identify which measured bits belong to x and which belong to y.
    """
    inputs = oracle.inputs
    inputs = [
        qubits
        for var_values in inputs.values()
        for qubits in var_values
    ]
    if not inputs:
        raise ValueError("need at least one qubit")

    qc = QuantumCircuit(oracle.circuit.num_qubits)
    # initialize with equal probability
    qc.h(inputs)

    phase_oracle = _phase_oracle(oracle)
    diffuser = _diffuser(qc.num_qubits, inputs)
    for _ in range(k):
        qc.compose(phase_oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    return qc, inputs

def _try_k(
    oracle: Oracle,
    pred: Bit,
    allowed: dict[str, set[int]],
    defaults: dict[str, int],
    k: int
) -> dict[str, int] | None:
    """
    run grover with k iterations and check the most frequent results

    return a valid solution, or None if none of the top results are solutions
    """
    # inputs: the sum of the variables' binary width
    circuit, inputs = _grover_circuit(oracle, k=k)
    # add classical bits to the circuit to store the results
    measured = ClassicalRegister(len(inputs), "measured")
    circuit.add_register(measured)
    for index, qubit in enumerate(inputs):
        circuit.measure(qubit, measured[index])

    # run the quantum simulation on a classical computer's CPU
    # scratch qubits just follow the inputs, so this method's cost tracks the inputs, not all 2^q states
    simulator = AerSimulator(method="matrix_product_state")
    execuble = transpile(circuit, simulator)
    result = simulator.run(execuble, shots=1024).result()
    counts = result.get_counts()

    # most frequent outcomes first: after amplification, solutions dominate
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)

    for bitstring, _ in ranked:
        bits = bitstring[::-1]
        measured_by_qubit = {}

        for position, qubit in enumerate(inputs):
            measured_by_qubit[qubit] = int(bits[position])
        assignment = defaults.copy()
        for name, qubits in oracle.inputs.items():
            value = 0
            # decode: turn bits in decimals
            for bit_index, qubit in enumerate(qubits):
                bit = measured_by_qubit[qubit]
                value += bit * (2 ** bit_index)
            assignment[name] = value

        if any(value not in allowed[name] for name, value in assignment.items()): 
            continue
        if evaluate(pred, assignment):
            return assignment

    return None

def search(
    params: Mapping[str, Iterable[int]],
    pred: Bit | UInt # expression tree for the condition
) -> dict[str, int] | None:
    """
    search for solutions in the space of params that satisfy pred
    the search domain is the cartesian product of each parameter's search space specified in params.
    return one solution set that satisfy the predicate
    """
    # preprocessing to make sure everything is in the right types and validated
    
    if not isinstance(params, Mapping): 
        raise TypeError("params must be mappings")
    # a UInt condition is true when it is non-zero
    if isinstance(pred, UInt): 
        pred = pred != 0
    validate_expression(pred)
    widths = _validate_widths(pred)
    allowed = _prepare_domains(params, widths)
    # empty set of possible parameters for one variable
    if any (not values for values in allowed.values()): 
        return None
    
    # deterministic allowed value for every param, in the case that 
    # the predicate is always true or always false (for inputs and return values)
    defaults = {
        name: min(values)
        for name, values in allowed.items()
    }
    simplified = cast(Bit, optimize_expr(pred))

    if simplified.bit_type == BitType.CONST: 
        expected = cast(bool, simplified.value)
        actual = bool(evaluate(pred, defaults))
        if actual != expected: 
            raise RuntimeError("optimized predicate disagrees with original predicate")
        return defaults if expected else None

    # generate new predicate based on allowed domain, then reoptimize
    effective = _restrict_domains(simplified, allowed)
    effective = cast(Bit, optimize_expr(effective))

    # compile_oracle only accepts decomposed logic; evaluate below keeps using pred
    oracle = compile_oracle(effective)
    # if compilation removes all remaining inputs
    if not oracle.inputs: 
        return defaults if evaluate(pred, defaults) else None

    num_input_qubits = sum(len(q) for q in oracle.inputs.values())
    root_n = math.sqrt(2 ** num_input_qubits)
    budget = math.ceil(BUDGET * root_n)
    limit = 1.0
    used = 0

    # BBHT: the number of solutions is unknown, so try random k under a growing limit
    while used < budget:
        remaining = budget - used

        choices = min(math.ceil(limit), remaining)
        k = random.randrange(choices)
        # k+1 makes sure 0 doesn't make zero progress
        used += k + 1
        assignment = _try_k(oracle, pred, allowed, defaults, k)
        if assignment is not None:
            return assignment
        # no search needs more than sqrt(N) iterations
        limit = min(limit * GROWTH, root_n)
    # budget used up, but this does not prove that it is impossible
    return None
