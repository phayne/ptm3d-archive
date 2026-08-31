"""Build a flux-cube layout before a run, without freezing the window.

`gui_core.flux_cubes.layout_plan` decides WHAT to build and where; this module
carries the decision out. It is separate from both because it is the only part
that needs Qt and a subprocess, and because the launch path and the card's own
button must run the same build -- a second implementation is a second set of
arguments, and the argument list is what the engine later refuses on.

⛔ NEVER ON THE GUI THREAD. The measured full-grid sweep on a 2.4 M-face site
mesh was 66 s. Sixty-six seconds of frozen window with no cancel is what the
worker and the progress dialog exist to prevent, and `run(...)` returns only
after the worker has finished or the user has stopped it.

⛔ A CANCELED OR FAILED BUILD LEAVES NO FILE. The tool writes its output as it
goes, so a terminated build can leave a partial `.fcg` on the request's own
path -- which is exactly the path the next run would look at, find, and reuse.
The header check in `cache_satisfies` would refuse a truncated one, but relying
on that means relying on a reader to catch a writer's mess; the partial file is
removed instead.
"""

import logging
import os
import subprocess
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

#: Wall-clock ceiling, the same one the card's button used. The measured
#: full-grid sweep on a 2.4 M-face site mesh was 66 s at 65,536 rays per patch;
#: this leaves room for a mesh several times larger and still ENDS rather than
#: hanging a launch forever.
TIMEOUT_S = 1800.0


class LayoutBuildWorker(QThread):
    """One `ptm_flux_cube_geom` run, off the GUI thread, emitting exactly once.

    `argv` is a LIST and the subprocess runs without a shell, so a path with a
    space or a quote in it is an argument and not syntax.

    The tool's own diagnostics are the message on failure, trimmed to the tail:
    it prints its measurements first and its refusal last, and the refusal is
    what the user has to read. A refusal is a refusal -- the tool aborts on a
    lattice point that misses the terrain, on a site at the pole, and on an
    unreadable occluder mesh, rather than dropping cubes -- so a non-zero exit
    means no layout was written.
    """

    #: One line of the tool's output, for the progress dialog's label.
    progress = pyqtSignal(str)
    #: (ok, message). Emitted exactly once, including on cancel and on crash.
    done = pyqtSignal(bool, str)

    def __init__(self, plan, cwd, parent=None):
        super().__init__(parent)
        self._plan = plan
        self._cwd = str(cwd)
        self._proc = None
        self._canceled = False

    def cancel(self):
        """Stop the tool. Safe to call from the GUI thread, and idempotent."""
        self._canceled = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:                                  # already gone
                pass

    def run(self):                                           # noqa: D102
        path = Path(self._plan.path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # The named-locations file the tool reads, written beside the
            # layout so the two travel together and a rebuilt layout is never
            # matched against somebody else's point list.
            if self._plan.locations_csv:
                Path(path.with_suffix(".locations.csv")).write_text(
                    self._plan.locations_csv)
            self._proc = subprocess.Popen(
                self._plan.argv, cwd=self._cwd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace",
                bufsize=1)
            tail = []
            for line in self._proc.stdout:
                line = line.rstrip()
                if line:
                    tail.append(line)
                    del tail[:-8]
                    self.progress.emit(line)
            self._proc.wait(timeout=TIMEOUT_S)
            if self._canceled:
                self._discard(path)
                self.done.emit(False, "The cube layout build was stopped, and "
                                      "no layout file was left behind.")
                return
            if self._proc.returncode != 0:
                self._discard(path)
                self.done.emit(False, "\n".join(tail) or
                               f"{Path(self._plan.argv[0]).name} exited "
                               f"{self._proc.returncode} without a message.")
                return
            if not path.exists():
                # rc == 0 IS NOT SUCCESS. The plan names the file the run will
                # be told to read; if it is not there, saying "built" would
                # hand the engine a path to nothing and cost the whole solve.
                self.done.emit(False, f"{Path(self._plan.argv[0]).name} "
                                      f"reported success and {path} was not "
                                      f"written.")
                return
            self.done.emit(True, "\n".join(tail))
        except subprocess.TimeoutExpired:
            self.cancel()
            self._discard(path)
            self.done.emit(False, f"The layout tool did not finish within "
                                  f"{TIMEOUT_S / 60:.0f} minutes and was "
                                  f"stopped.")
        except Exception as exc:                             # noqa: BLE001
            self._discard(path)
            self.done.emit(False, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _discard(path):
        """Remove a partial layout. Never raises -- this runs on a failure
        path, and a failure to clean up must not replace the real message."""
        try:
            if path.exists():
                os.remove(path)
        except OSError as exc:                               # noqa: BLE001
            logger.warning("[flux cubes] the partial layout %s could not be "
                           "removed (%s); it will be refused on its header "
                           "rather than reused.", path, exc)


def build_layout(parent, plan, cwd):
    """Build `plan`'s layout with a cancelable progress dialog. (ok, message).

    Returns only when the build has finished, failed or been stopped -- the
    caller is a launch sequence and has to know which -- but the window keeps
    painting throughout, because the waiting is done in a nested event loop
    rather than by blocking.
    """
    from PyQt6.QtCore import QEventLoop, Qt
    from PyQt6.QtWidgets import QProgressDialog

    dialog = QProgressDialog(
        f"Building the cube layout: {plan.request.describe()}.\n"
        f"It measures the mesh and casts rays from every cube face.",
        "Stop", 0, 0, parent)
    dialog.setWindowTitle("Flux cubes")
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    # No auto-close and no auto-reset: this dialog is closed by the result,
    # not by a value it never receives (the tool reports no percentage).
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setMinimumDuration(0)

    loop = QEventLoop()
    outcome = {}
    worker = LayoutBuildWorker(plan, cwd)

    def _finished(ok, message):
        outcome["ok"] = ok
        outcome["message"] = message
        loop.quit()

    worker.done.connect(_finished)
    worker.progress.connect(
        lambda line: dialog.setLabelText(
            f"Building the cube layout: {plan.request.describe()}.\n{line}"))
    dialog.canceled.connect(worker.cancel)
    worker.start()
    dialog.show()
    loop.exec()
    # The thread has emitted, but has not necessarily returned from run();
    # wait for it so the process is reaped before the caller moves on.
    worker.wait(5000)
    dialog.close()
    return bool(outcome.get("ok")), str(outcome.get("message", ""))


def prepare_from_report(parent, report, config_manager, project_root,
                        info, error):
    """Carry out a launch check's `flux_cube_plan`. True to go ahead.

    ONE FUNCTION, TWO LAUNCHERS. The main window and the dashboard's thermal
    module both call `run_preflight` and both start the engine, and a launcher
    that ignored the plan would start a run with the cubes on and no layout --
    which now passes the launch check by design, solves to the end, and refuses
    in the post-solve pass. That is the whole cost this feature exists to
    avoid, so the step is shared rather than written twice.

    `info` and `error` are the caller's own console routers, because the two
    launchers tag their lines differently and a shared function should not pick
    one of the two prefixes.

    Both outcomes end with `flux_cubes.cache_file` naming the layout that will
    run, which is what puts it in the run YAML:

    * REUSE -- a layout for exactly this placement is on disk. It was matched
      on its own header (spacing, radius, face count, named points), not on its
      name, so a layout for a different placement cannot be picked up.
    * BUILD -- `ptm_flux_cube_geom` runs in a worker with a cancelable progress
      dialog. A failed or stopped build leaves no file and does NOT start the
      run.
    """
    from PyQt6.QtWidgets import QMessageBox
    from gui_core import flux_cubes as fc

    plan = getattr(report, "flux_cube_plan", None)
    if plan is None:
        return True
    if plan.action == fc.PLAN_REUSE:
        info(f"reusing {plan.path} -- {plan.reason}")
    else:
        info(f"{plan.reason}. Building {plan.request.describe()} into "
             f"{plan.path}")
        ok, message = build_layout(parent, plan, str(project_root))
        if not ok:
            error(f"the layout was not built: {message}")
            QMessageBox.critical(
                parent, "Flux cubes",
                "The cube layout could not be built, so the run was not "
                "started.\n\n" + message + "\n\nNothing else was changed. A "
                "run without its layout would solve to the end and then stop "
                "in the flux-cube pass, which is why this is a refusal rather "
                "than a warning.")
            return False
        for line in (message or "").splitlines():
            info(line)
        info(f"layout written: {plan.path}")
    config_manager.set_value("flux_cubes.cache_file", plan.path)
    return True
