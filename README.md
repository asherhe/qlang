# qlang

Build symbolic Boolean and unsigned-integer expressions, evaluate them classically,
and compile supported predicates into quantum phase oracles. Grover search finds
sampled satisfying assignments. QAOA minimizes a symbolic cost over sampled
feasible assignments using a local statevector simulator.

## Quickstart

Use Python 3.14 or newer. Run these commands from the repository root.

Install the locked dependencies and the editable package.

```bash
uv sync --frozen
```

Run the basic example, then either algorithm example.

```bash
uv run examples/run_basic.py
uv run examples/run.py
uv run examples/run_qaoa.py
```

| Example | What it runs |
| --- | --- |
| `examples/run_basic.py` | Expression construction, evaluation, simplification, and oracle compilation |
| `examples/run.py` | Grover search for `x + y == 3` |
| `examples/run_qaoa.py` | QAOA with the same constraint and the objective `2*x + y` |

Examples are standalone scripts and are outside pytest's configured discovery
path. Algorithm results are sampled, so assignments can vary between runs.

## Expressions and classical evaluation

```python
import qlang

qlang.ASTNode.reset()
x = qlang.uint("x", width=2)
y = qlang.uint("y", width=2)
constraint = x + y == 3
objective = 2 * x + y

assignment = {"x": 1, "y": 2}
assert qlang.evaluate(constraint, assignment) is True
assert qlang.evaluate(objective, assignment) == 4
```

`uint("x", width=2)` declares a variable with values from 0 to 3. `uint(3)`
creates a constant with inferred width. `var_bit("p")` declares a Boolean
variable. `bits_uint((low_bit, high_bit))` assembles an integer with its lowest
bit first. Integer comparisons produce symbolic `Bit` predicates.

Use `&`, `|`, `^`, and `~` to compose Boolean predicates. Parenthesize comparisons,
for example `(x < 2) & (y != 0)`. Python `and`, `or`, `not`, and `if constraint`
attempt to convert a symbolic expression to a Python Boolean and raise an error.
Call `evaluate(constraint, assignment)` to obtain the Boolean value.

Provide all referenced variable names in an assignment. A `Bit` accepts 0/1
or Boolean values. A UInt value must fit its declared unsigned width.

### Width and bit semantics

| Operation | Result width and classical meaning |
| --- | --- |
| Variable of width `w` | Values from 0 through `2**w - 1` |
| `a + b` | `max(a.width, b.width) + 1`, preserving the carry |
| `a - b` | `max(a.width, b.width)`, wrapping modulo `2**width` |
| `a * b` | `a.width + b.width` |
| `a << k` | Keeps `a.width` and discards overflowing bits |
| `a >> k` | Keeps `a.width` and shifts in zeros |
| `a.bits[i]` | Bit `i`, with index 0 the least significant bit |

Subtraction represents unsigned wrapping arithmetic. For example, subtracting 1
from a two-bit zero produces 3. Explicit constants must fit their requested width.

### Expression lifecycle

The expression cache and named-variable registry are shared across the process.
Call `ASTNode.reset()` before constructing each independent problem, including
when rerunning notebook cells with different variable widths. Within a problem,
reuse the variable objects you already created. A conflicting declaration with
the same name raises an error.

Reset at the problem boundary, before constructing its expressions. Keep that
problem's variables together through evaluation, compilation, and solving.

## Simplification and oracle compilation

```python
simplified = qlang.optimize_expr(x + 0)
assert qlang.evaluate(simplified, assignment) == 1

oracle = qlang.compile_oracle(constraint)
print(oracle.inputs)                # Variable names mapped to input qubit indices.
print(oracle.circuit.num_qubits)    # Includes workspace qubits.
```

This snippet continues the expression example above. `optimize_expr()` rewrites
expressions while preserving their evaluated meaning. `compile_oracle()` accepts
a `Bit` predicate and performs optimization internally. Its circuit marks
satisfying basis states with a minus sign and restores workspace to zero.
`oracle.output` is empty for this phase-oracle representation. Each register in
`oracle.inputs` lists its qubits from least significant bit to most significant.

The compiler currently has gates for Boolean AND, OR, XOR, NOT, equality and
greater-or-equal comparison, plus integer addition and subtraction. Other
comparisons are rewritten through supported comparisons. Constant shifts can
be rewritten as bit wiring. General multiplication and variable-controlled shifts
have no compiler gadget, although classical evaluation supports them. Some
constant multiplications simplify into supported expressions.

Selecting bits from computed integers remains limited by compiler input
validation. Expressions inside `BITS` wrappers must reduce to supported compiler
operations after optimization. Validate a new construction on a small problem
before using it in an algorithm. Classical evaluability alone does not establish
oracle compiler support.

## Grover search

```python
solution = qlang.search({"x": range(4), "y": range(4)}, constraint)
if solution is not None:
    assert qlang.evaluate(constraint, solution)
```

The implementation uses local Qiskit Aer simulation, repeated Grover trials,
and 1,024 measurement shots per trial. A returned assignment passes the original
predicate and the post-measurement domain filter. `None` means the search budget
ended without accepting an assignment. It does not prove infeasibility.

Use a domain for every referenced variable, keep values within its declared
width, and avoid extra parameters. Current limitations include constant
predicates, variables removed by simplification, incomplete domain handling, and
consideration of only the three most frequent outcomes per trial. Domain
restrictions are applied after measurement. The randomized iteration schedule
also differs from the original BBHT schedule and has no stated confidence bound.

## QAOA with a symbolic objective

```python
solution = qlang.qaoa(
    [x, y],
    constraint,
    objective=objective,
    penalty=10.0,
    iterations=16,
)
if solution is not None:
    print(solution, qlang.evaluate(objective, solution))
```

This continues the same four-qubit problem. Supply a symbolic `UInt` objective.
Omit `objective` for a feasibility-only problem. Pass every variable referenced
by the constraint or objective in the variable list, once each. Its order defines
the register layout used both for cost compilation and decoding.

The circuit has one cost/mixer layer. `iterations` is the positive integer number
of grid values per angle, so `iterations=16` evaluates 256 angle pairs over
`[0, 2*pi)`. Each evaluation computes an exact statevector mean cost. The chosen
angles are then sampled for 1,024 shots. With an objective, QAOA returns the
lowest-cost feasible assignment observed in those shots. Without one, it returns
an observed feasible assignment.

`None` means no feasible assignment was sampled. A returned assignment is not
guaranteed to attain the global minimum. For this example, the exact optimum is
`{"x": 0, "y": 3}` with cost 3, since the objective becomes `3 + x` under the
constraint. The example prints the observed cost and the known minimum separately.

### Direct compilation and the general fallback

For a linear equality, the direct path constructs
`objective + penalty * (left - right)**2` algebraically. It supports variables,
constants, addition, and multiplication by a constant on either side of the
equality. A supplied objective must also have one of those linear forms. This
path rejects a nonlinear objective. Subtraction and shifts are outside the
current linear extractor.

For `n` input bits, a linear equality produces at most quadratically many
identity, Z, and ZZ terms. If the constraint is outside the direct path, the
general fallback evaluates every assignment and constructs the Pauli expansion
of `objective + penalty * violation`, where violation is 0 or 1. The fallback
can handle nonlinear objectives supported by `evaluate()`, but its coefficient
construction takes roughly `4**n` work and may produce `2**n` terms.

There is currently **no enforced fallback qubit limit**. Start with the four-bit
examples and increase sizes cautiously. Statevector storage itself grows as
`2**n` even for the direct path, and the grid evaluates `iterations**2` circuits.
A faster cost compiler does not remove those simulation costs or provide a
general optimization speedup guarantee.

### Choosing a penalty

Use a positive penalty. For integer linear equalities, every violation has a
squared residual of at least 1. A penalty larger than the objective's total
range makes every infeasible assignment cost more than every feasible one,
provided the feasible set is nonempty. The same sufficient bound applies to the
fallback's unit violation penalty.

In the example, `2*x + y` ranges from 0 to 9, so `penalty=10` satisfies that bound.
If the constraint is satisfiable, the combined cost then has a feasible global
minimum. Finite-depth QAOA and finite sampling can still miss that minimum.
Larger penalties also change the angle landscape, so the coarse grid can require
retuning. The API does not currently choose a penalty automatically.

## Module guide

| Module | Responsibility |
| --- | --- |
| `expression.py` | Symbolic nodes, variable factories, bit ordering, and problem reset |
| `evaluate.py` | Classical evaluation for a supplied assignment |
| `optimize.py` | Expression decomposition and simplification |
| `compile/` | Gate construction, qubit management, and clean phase oracles |
| `qalgorithm.py` | Grover circuit construction and sampled search |
| `minimize.py` | Cost operators, one-layer QAOA, and sampled objective minimization |

The public entry points above are exported through `qlang`.

## Tests

```bash
uv run pytest
uv run pytest tests/qlang/test_minimize.py
```

Strict xfails record known limitations. An unexpected pass means the marker needs
review. Read the reported failures and skips when assessing support for an input
form; the examples demonstrate specific small problems.

## Core development workflow

### Running code

Run any script in the project context.

```bash
uv run python path/to/script.py
```

### Lockfile usage

`uv sync` reads `uv.lock` and installs the exact pinned versions recorded in
it. The lockfile is reproducible across machines and environments.

Regenerate the lockfile after changing dependencies.

```bash
uv lock
```

Install exactly what the lockfile specifies (fresh checkout).

```bash
uv sync --frozen
```

Update dependencies and refresh the lockfile.

```bash
uv lock --upgrade
```
