"""Helpers for examples/qlang_demo.ipynb, kept here so the notebook stays readable."""

import math
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from qiskit import ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from .compile import compile_oracle
from .evaluate import evaluate
from .expression import ASTNode, uint
from .qalgorithm import BUDGET, GROWTH, _grover_circuit, _try_k

# x is 3 bits (0-7) and y is 2 bits (0-3)
PARAMS = {"x": range(8), "y": range(4)}
INPUTS = [{"x": x, "y": y} for x in PARAMS["x"] for y in PARAMS["y"]]

SHOTS = 1024
SOLUTION_COLOR = "#2a78d6"
OTHER_COLOR = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"


def fresh_variables():
    """Create x and y. Variables can only be defined once, so clear them first."""
    ASTNode.reset()
    return uint("x", 3), uint("y", 2)


def find_solutions(pred):
    """Try every input on a normal computer."""
    return [assignment for assignment in INPUTS if evaluate(pred, assignment)]


def print_truth_table(pred):
    """Run every input through the oracle and compare it with the classical answer."""
    oracle = compile_oracle(pred)
    matches = 0

    print(" x  y  classical  oracle sign")
    for assignment in INPUTS:
        basis = sum(
            ((assignment[name] >> bit) & 1) << qubit
            for name, qubits in oracle.inputs.items()
            for bit, qubit in enumerate(qubits)
        )
        state = Statevector.from_int(basis, 2 ** oracle.circuit.num_qubits).evolve(oracle.circuit)
        sign = round(state.data[basis].real)
        expected = bool(evaluate(pred, assignment))
        matches += (sign == -1) == expected
        if expected:
            print(f"{assignment['x']:2} {assignment['y']:2}  {str(expected):9}  {sign:+d}")

    print(f"\n{matches} of {len(INPUTS)} inputs match (only solutions are listed)")


def _measure(oracle, k):
    """Run Grover for k iterations and count how often each (x, y) is measured."""
    circuit, qubits = _grover_circuit(oracle, k=k)
    measured = ClassicalRegister(len(qubits), "measured")
    circuit.add_register(measured)
    for index, qubit in enumerate(qubits):
        circuit.measure(qubit, measured[index])

    simulator = AerSimulator(method="matrix_product_state")
    counts = simulator.run(transpile(circuit, simulator), shots=SHOTS).result().get_counts()

    totals = {}
    for bitstring, count in counts.items():
        bits = dict(zip(qubits, map(int, bitstring[::-1])))
        decoded = tuple(
            sum(bits[qubit] << i for i, qubit in enumerate(oracle.inputs[name]))
            for name in ("x", "y")
        )
        totals[decoded] = totals.get(decoded, 0) + count
    return totals


def plot_amplification(pred, title, ks=(0, 1, 2)):
    """Chart how often each input is measured after k Grover iterations."""
    oracle = compile_oracle(pred)

    # solutions first, so they sit together on the left of each chart
    ordered = sorted(INPUTS, key=lambda i: (not evaluate(pred, i), i["x"], i["y"]))
    pairs = [(i["x"], i["y"]) for i in ordered]
    is_solution = [bool(evaluate(pred, i)) for i in ordered]
    colors = [SOLUTION_COLOR if solution else OTHER_COLOR for solution in is_solution]

    fig, axes = plt.subplots(1, len(ks), figsize=(13, 4), sharey=True, facecolor=SURFACE)
    for ax, k in zip(axes, ks):
        totals = _measure(oracle, k)
        percent = [100 * totals.get(pair, 0) / SHOTS for pair in pairs]
        share = sum(p for p, solution in zip(percent, is_solution) if solution)

        ax.bar(range(len(pairs)), percent, color=colors, width=0.8)
        ax.set_facecolor(SURFACE)
        ax.set_title(f"k = {k}: {share:.0f}% of shots are solutions", color=TEXT, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_xlabel(f"All {len(pairs)} inputs, with the {sum(is_solution)} solutions on the left", color=MUTED, fontsize=10)
        ax.tick_params(axis="y", colors=MUTED, length=0)
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")

    axes[0].set_ylabel("Share of shots (%)", color=MUTED)
    handles = [Rectangle((0, 0), 1, 1, color=color) for color in (SOLUTION_COLOR, OTHER_COLOR)]
    fig.legend(handles, ["Solution", "Not a solution"], loc="upper right", frameon=False, ncol=2, labelcolor=TEXT)
    fig.suptitle(title, color=TEXT, fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def traced_search(pred):
    """The same BBHT loop as qlang.search, printing each round."""
    oracle = compile_oracle(pred)
    allowed = {name: set(values) for name, values in PARAMS.items()}
    root_n = math.sqrt(2 ** sum(len(q) for q in oracle.inputs.values()))
    limit, used, round_number = 1.0, 0, 0

    while used < BUDGET * root_n:
        round_number += 1
        k = random.randint(1, math.ceil(limit))
        used += k
        result = _try_k(oracle, pred, allowed, k)
        print(f"round {round_number:2}  limit {limit:4.2f}  k = {k}  total iterations {used:2}  found: {result}")
        if result is not None:
            return result
        limit = min(limit * GROWTH, root_n)

    print(f"budget of {BUDGET * root_n:.0f} iterations used up: most likely no solution")
    return None
