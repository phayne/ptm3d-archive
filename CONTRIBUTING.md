# Contributing to PTM3D

Short version of the two things that are easy to get wrong.

## Changing an engine configuration key

The engine's YAML parser is the only authority on which configuration keys exist
and what each one defaults to (`src/cpp/tools/config_parser.cpp`, with the
struct defaults in `config_parser.h`). The Python layer keeps a committed copy
of that table in `src/python/ptm/engine_defaults.py`, and the two must agree.

**If you add a key, remove a key, or change a default, regenerate the table and
commit the result:**

```bash
python3 -m ptm.engine_config --regenerate
git add src/python/ptm/engine_defaults.py
```

Changing an existing default is the case worth spelling out, because it does not
look like a change to the configuration surface. Nothing in the YAML files
moves, no key appears or disappears, and the engine goes on accepting every
configuration it accepted before — it simply runs a different simulation when
the key is absent. The committed table would go on reporting the old value to
anything that asked.

Two properties of the regenerator are worth knowing:

- **It reads the C++ sources, not a build.** No compiler, no CSPICE, no CMake
  step. It works in a checkout that has never been built.
- **It reads *your* checkout.** Regenerating on a tree that does not contain the
  C++ change produces a table that is stale in a way that looks current. Make
  the C++ edit first, then regenerate.

`src/python/tests/test_config_parity.py` re-derives the table from the C++
sources on every run and fails if it no longer matches the committed one, so a
missed regeneration is a red CI job rather than a Python layer that quietly
believes the old value. It also fails when the extractor stops recognizing the
parser's shape, and when a shipped example configuration grows a key the engine
does not read.

Run it directly — it needs no pytest:

```bash
python3 src/python/tests/test_config_parity.py
```

One of its five checks reads the GUI's configuration writer and needs PyQt6. It
reports `SKIP` where PyQt6 is absent, and the summary counts skipped separately
from failed. CI installs PyQt6 and runs all five.

## The Python layer

`pip install -e .` from the repository root installs the `ptm` package and the
`ptm` command. Nothing in it computes physics: it reads what the engine wrote
and it starts the engine.

Before opening a pull request that touches `src/python/ptm/` or `gui/`:

```bash
python3 src/python/tests/test_config_parity.py
python3 src/python/tests/test_results_api.py
python3 src/python/tests/test_engine_process.py
```

These three are what the `python-layer` CI job runs, and they need no engine
binary, no SPICE kernels and no GPU.
