# qlang unit tests

Tests are grouped by source module. Run from the repository root with the
project environment (after installing the project dependencies):

```powershell
uv pip install --python .venv/Scripts/python.exe -r src/qlang/tests/requirements.txt
.venv/Scripts/python.exe -B -m pytest src/qlang/tests
```

From `src/qlang`, use `python -B -m pytest tests` with the project environment
activated. The test-local `pytest.ini` limits discovery to this suite, enables
expected-failure summaries, and makes unexpected passes fail the run.

## Coverage

- `test_evaluate.py`: constants, Boolean truth tables and comparisons, all UInt
  comparisons, nested arithmetic, widening, fixed-width shifts and two's
  complement subtraction, reflected operators, indexed bits, input validation,
  input immutability, malformed operand counts, and per-evaluation memoization.
- `test_expression.py`: public factories, bit ordering, inferred and explicit
  widths, symbolic operator construction, reflected operand order, forbidden
  truth testing, and identity-based expression caching.
- `test_optimize.py`: decomposition, constant folding, Boolean identities,
  comparison bounds, arithmetic identities, reassociation, shifts, subtraction,
  carry preservation, width-sensitive parent expressions, source-tree
  immutability, rule registration/precedence, and shared-node traversal.
- `test_compile.py`: Oracle metadata, independent ancilla defaults, output
  modes, and the compiler's currently unfinished return contract.
- `test_qalgorithm.py`: two small specification cases for unfinished search.
- `test_api.py`: existing public package exports.

The suite uses representative boundaries and complete small Boolean domains,
not random inputs or large exhaustive integer searches. Optimizer equivalence
checks use the independently tested evaluator; focused regressions also assert
explicit expected values.

## Expected failures

Every marked case was first observed failing against the current source.
Markers have specific reasons, use strict mode, and usually restrict the
exception type. They preserve the intended assertion rather than accepting the
buggy result. Once a bug is fixed, remove its marker; an unexpected pass reports
that the regression is ready to become a normal passing test.

Current groups: index-zero rejection in public UInt factories; zero parent-width
validation; missing parent-width error ordering; Bit comparison optimization;
constant subtraction; overflowing constant left shifts; carry loss in `x+x`;
width-sensitive constant folding; and unfinished compile/search entry points.

`conftest.py` constructs valid UInt AST fixtures directly because the public
factory currently rejects every variable UInt at bit index zero. Public factory
behavior is tested separately, without monkeypatching production behavior.
The shared expression cache is isolated and restored around each test.

Source modules, project dependencies, and the lockfile are not changed by this
suite. Compiler/search expected failures are specification placeholders, not
quantum circuit or algorithm correctness tests.
