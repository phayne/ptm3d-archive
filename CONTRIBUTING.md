# Contributing to PTM3D

Short version of the three things that are easy to get wrong.

## American English

PTM3D is written in American English — documentation, tutorials, code comments,
docstrings, and anything a user can see, including GUI labels and command-line
help.
<!-- american-english: allow-begin (this section has to quote what it forbids) -->
`color`, not `colour`; `behavior`, not `behaviour`; `normalize`, not
`normalise`; `meter`, not `metre`; `gray`, not `grey`.
<!-- american-english: allow-end -->

```bash
python3 scripts/check_american_english.py
```

CI runs this on every push. Three kinds of hit are legitimate and stay British:

- **Quoted text you do not own** — a paper title in a citation, third-party
  license text, vendored code.
<!-- american-english: allow-begin -->
- **External format literals** — the OGC WKT unit name is spelled `metre`, and
  GDAL parses that string.
<!-- american-english: allow-end -->
- **Code identifiers** — a function, attribute, config key or saved-data field
  whose British name cannot change without breaking callers or existing runs.

Record such a case in the `ALLOW` table at the top of
`scripts/check_american_english.py`, which pairs a path with the substring that
makes the line legitimate, or put `american-english: allow` at the end of the
line. A run of lines — prose that has to quote the spellings it forbids, as this
section does — goes between `american-english: allow-begin` and
`american-english: allow-end` in a comment. Do not widen a rule to silence one
hit.

Note two words that are already correct and must never be "fixed":
`cancellation` keeps both L's in American usage (only `canceled` and
`canceling` drop one), and `analyses` is the ordinary plural of `analysis`.

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
