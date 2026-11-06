# qlang

Write rules and costs as Python expressions, compile them into quantum circuits,
and explore how quantum search behaves. qlang combines a symbolic expression
library, Grover search, QAOA optimization, and an interactive notebook showing
the expression graph, circuit, and solution heatmap.

## Get started

Use Python 3.10 or newer. Run these commands from the repository root.

Install the locked dependencies and the editable package.

```bash
uv sync --frozen
uv run examples/run_basic.py
```

To install the library directly from GitHub instead:

```bash
python -m pip install "qlang @ git+https://github.com/asherhe/qlang.git"
```

Open **[examples/qlang.ipynb](examples/qlang.ipynb)** in Jupyter or VS Code's
Jupyter extension and choose **Run All**. Its setup installs dependencies into
the selected kernel. Restart the kernel after a first-time widget installation
if controls do not appear.

The notebook has editable variables, original/optimized expression graphs,
full-expression labels, a circuit schematic, and a Grover heatmap. It shares
one expression model across all three views. The visualization code is bundled
inside a collapsed cell, so the notebook needs no companion helper files;
enable `SHOW_IMPLEMENTATION` to inspect it.

Heatmaps update live using classical evaluation or ideal Grover predictions.
**Run Grover** samples the compiled circuit on demand. Live heatmaps support up
to 8 input bits; simulation supports up to 18 total qubits, including workspace.
Domains filter the heatmap display; they do not change its oracle, and displayed
probabilities are not renormalized after filtering.

## Constraints and objectives

A **constraint** is a rule that an acceptable answer must satisfy. An
**objective** is a number used to rank acceptable answers; qlang minimizes it.
Finding any acceptable answer is **feasibility**; finding one with the lowest
cost is **optimization**.

For example, choose two quantities `x` and `y` totaling three units. Each unit
of `x` costs two points and each unit of `y` costs one:

```python
import qlang

qlang.ASTNode.reset()  # Start an independent problem.
x = qlang.uint("x", width=2)  # Two bits: values 0 through 3.
y = qlang.uint("y", width=2)

constraint = x + y == 3
objective = 2 * x + y

assignment = {"x": 1, "y": 2}
assert qlang.evaluate(constraint, assignment) is True
assert qlang.evaluate(objective, assignment) == 4

oracle = qlang.compile_oracle(constraint)
print(oracle.inputs)              # Variable names mapped to input qubits.
print(oracle.circuit.num_qubits)  # Input qubits plus workspace.
```

`{"x": 1, "y": 2}` is feasible, with cost 4. The best feasible assignment is
`{"x": 0, "y": 3}`, with cost 3. The constraint determines what is allowed;
the objective determines what is preferred.

### Find an answer with Grover

Continuing the example:

```python
solution = qlang.search({"x": range(4), "y": range(4)}, constraint)
if solution is not None:
    assert qlang.evaluate(constraint, solution)
```

Grover seeks a satisfying assignment; it does not minimize the objective.
Supply a domain for every referenced variable and keep its values within the
declared width. The current implementation uses local Qiskit Aer simulation,
1,024 shots per trial, and validates sampled candidates against the predicate
and domains. Domain restrictions are currently checked after measurement.

### Prefer lower costs with QAOA

```python
solution = qlang.qaoa(
    [x, y], constraint,
    objective=objective,
    penalty=10.0,
    iterations=16,
)
if solution is not None:
    assert qlang.evaluate(constraint, solution)
    print(solution, qlang.evaluate(objective, solution))
```

QAOA (Quantum Approximate Optimization Algorithm) searches a parameterized
quantum circuit for low-cost answers. A **penalty** adds cost to constraint
violations. Here the objective ranges from 0 to 9, so a penalty of 10 exceeds
that range and makes a violated integer equality more costly than any feasible
answer. The API does not choose a penalty automatically. Omit `objective` for
a feasibility-only problem; include every referenced variable in the list once.

This implementation has one cost/mixer layer. `iterations=16` tries 16 values
for each of two angles (256 pairs), then samples 1,024 outcomes and returns the
lowest-cost feasible answer observed. It does not guarantee the global optimum.
For either solver, `None` means no answer was returned within the run; it does
not generally prove that the problem has no solution.

## Expression rules and limits

- `var_bit("p")` creates a Boolean variable; `uint("x", width=w)` creates an
  unsigned integer from 0 through `2**w - 1`. `uint(3)` creates a constant.
- Build Boolean logic with `&`, `|`, `^`, and `~`. Parenthesize comparisons:
  `(x < 2) & (y != 0)`. Python `and`, `or`, `not`, and `if constraint` cannot
  evaluate symbolic expressions; use `qlang.evaluate()` with an assignment.
- Addition widens by one bit beyond the larger operand; multiplication uses
  the sum of operand widths. Subtraction keeps the larger width and wraps:
  two-bit `0 - 1` is 3. Shifts keep the left operand's width and discard overflow.
  Bit index 0 is the least significant bit.
- Call `ASTNode.reset()` before each independent problem. The variable registry
  and expression cache are process-wide; reuse variable objects within a problem.
- `optimize_expr()` simplifies an expression. `compile_circuit()` produces a
  forward reversible computation with output qubits; workspace may retain
  intermediate values. `compile_oracle()` accepts a Boolean predicate, marks
  satisfying inputs by phase, and reverses the computation to clean workspace.
- Classical evaluation supports more expressions than circuit compilation.
  General variable-by-variable multiplication and variable-controlled shifts
  have no compiler gadget. Some constant arithmetic simplifies into supported
  gates; test new constructions on small inputs.
- QAOA builds costs directly for supported linear equalities and linear
  objectives. Its general fallback enumerates assignments and takes roughly
  `4**n` coefficient-construction work for `n` input bits, with no enforced
  fallback limit. Statevector simulation also grows exponentially. These demos
  do not establish a practical quantum speedup.

## Examples and development

| File                                                   | Purpose                                                  |
| ------------------------------------------------------ | -------------------------------------------------------- |
| [examples/qlang.ipynb](examples/qlang.ipynb)           | Interactive AST, circuit, and Grover heatmap             |
| [examples/qlang_demo.ipynb](examples/qlang_demo.ipynb) | Guided walkthrough of the library and algorithms         |
| [examples/run_basic.py](examples/run_basic.py)         | Expressions, evaluation, simplification, and compilation |
| [examples/run.py](examples/run.py)                     | Grover search                                            |
| [examples/run_qaoa.py](examples/run_qaoa.py)           | QAOA with a constraint and objective                     |

```bash
uv run examples/run.py
uv run examples/run_qaoa.py
uv run pytest
```

The implementation lives in [src/qlang](src/qlang): `expression.py`,
`evaluate.py`, `optimize.py`, `compile/`, and `qalgorithm/`. Strict expected
failures in the test suite record known limitations; review the reported results.
