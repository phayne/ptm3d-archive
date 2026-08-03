# Private tools (not part of the public PTM3D distribution)

These modules were removed from the public/colleague GUI build and staged here
so they can later be spun into **separate, standalone GUIs**. They are kept
intact and recoverable on purpose; they are **not** deleted.

This directory is excluded from `git archive` distribution tarballs via the
repository `.gitattributes` (`gui/_private_tools/ export-ignore`). Do not wire
anything here back into `ptm3d_gui.py` or `dashboard.py` if you want it to stay
out of the public build.

## Contents

- `batch/` — the A3GT batch-runs dashboard panel (`batch_dashboard_panel.py`).
  Hardcodes Paul's A3GT campaign paths (`~/Research/moon/artemis/a3gt`,
  `run_a3gt_batch.py`, `A3GT_MM01-15`). Was a tab in `ptm3d_gui.py`.
- `exosphere/` — the Europa exosphere GUI (`exosphere_gui.py` +
  `run_exosphere_gui.py` + `exo_core/` + `exo_panels/`) and its dashboard
  adapter (`exosphere_adapter.py`, `exosphere_analysis.py`). Was the `EXO`
  module of `dashboard.py`.
- `viewshed/` — the PEGASUS viewshed GUI (`viewshed_gui.py` +
  `run_viewshed_gui.py` + `viewshed_core/` + `viewshed_panels/`) and its
  dashboard adapter (`viewshed_adapter.py`, `viewshed_analysis.py`). Was the
  `VIS` module of `dashboard.py`.

## Running the standalone GUIs

The exosphere and viewshed GUIs still import shared code from the parent `gui/`
package (`gui_core`, `gui_panels`, `dashboard_core`). Their launchers were
patched to put **both** their own tool directory and the parent `gui/`
directory on `sys.path`, so they run in place:

```bash
python gui/_private_tools/exosphere/run_exosphere_gui.py
python gui/_private_tools/viewshed/run_viewshed_gui.py
```

## Known caveats for future standalone extraction

- The dashboard **adapters** (`*_adapter.py`) still contain
  `from module_adapters.<tool>_analysis import ...`. That import path no longer
  resolves after the move; it is already `try/except`-guarded, so the adapter
  degrades gracefully (its analysis panel is simply disabled) rather than
  crashing. Fix the import path when reviving an adapter for a dashboard.
- These adapters wrap the tools for the (demoted) unified dashboard, not for a
  standalone app; a standalone GUI uses the `*_gui.py` entry point directly and
  does not need the adapter.
