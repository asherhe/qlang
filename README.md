# qlang

Compile expressions into quantum circuits!

## Examples

Exploratory and demo scripts live in the `examples/` directory. They are
meant for manual execution and are not part of the installed package.

Run any example with `uv run`, which executes it in the project's
virtual environment:

```bash
uv run examples/run.py
```

## Quickstart

Create and sync the virtual environment. This also installs `qlang` itself
as an editable package.

```bash
uv sync
```

Run the basic example to verify the package imports.

```bash
uv run examples/run_basic.py
```

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
