"""Launching ``ptm_thermal_sim`` and reading what it says while it runs.

Two entry points, for the two ways people use this:

* :func:`run_engine` -- one blocking call, returns a :class:`CompletedRun`.
  This is what a script or a notebook wants, and what everything in the tree
  was hand-rolling on top of ``EngineProcess``.
* :class:`EngineProcess` -- the callback-shaped driver underneath, for a caller
  with its own event loop (the GUI wraps it in a ``QThread``).

This is the process-driving half of ``gui/gui_core/simulation_runner.py``,
extracted with no Qt so that a script, a notebook or the GUI can all use the
same driver. The GUI's ``SimulationThread`` is now a ``QThread`` wrapper that
forwards these callbacks to its signals; the behavior below is the behavior it
had, moved rather than rewritten.

What it carries, and why each piece exists:

* **Severity classification by content, not by stream.** The engine writes
  informational and warning banners to stderr during successful runs
  (``[eq-seed] ...``, ``[machine-profile] ... CONSERVATIVE defaults``,
  ``[GPU Thermal] dispatch self-chunk ceiling ...``), so treating stderr as
  failure mislabels a good run. :func:`classify_severity` reads both streams
  and decides on the text.
* **A CPU-clock stall watchdog.** A process blocked reading a dehydrated
  cloud-synced file accumulates no CPU time and is otherwise
  indistinguishable from a long computation. The child's CPU clock is sampled
  every ``WATCHDOG_INTERVAL_S`` and a stall is reported through
  ``on_heartbeat``.
* **Process-group teardown.** The child runs in its own session, so
  :meth:`EngineProcess.stop` signals the group: TERM, then KILL after 5 s,
  with bounded waits so a stuck child never hangs the caller.

No physics: this module starts a process, reads its output and reports what it
saw.
"""

from __future__ import annotations

import datetime
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import paths

__all__ = [
    "BinaryProvenance",
    "CompletedRun",
    "EngineProcess",
    "binary_provenance",
    "build_command",
    "classify_severity",
    "engine_flags",
    "engine_supports_flag",
    "extract_progress",
    "file_md5",
    "find_engine_executable",
    "resolve_engine",
    "run_engine",
    "sample_cputime",
]

find_engine_executable = paths.find_engine_executable


# --- Severity classification ------------------------------------------------
# Genuine-failure signatures (error) are kept narrow so ordinary banners are
# never reported as errors; a real failure also carries a non-zero exit code,
# which the caller sees independently.
#
# The engine spells a fatal in three ways, and all three must match:
#     FATAL: Failed to load mesh from ...
#     FATAL [gpu]: GPU requested but this binary is CPU-only
#     [FATAL] Caught signal 11
# The bracketed tag forms are the ones the startup guards use -- [gpu],
# [mesh-units], [coupling], [exchange] -- so the optional tag and the closing
# bracket below are not decoration.
_ERROR_RE = re.compile(
    r'(?<![A-Za-z])(?:FATAL|ERROR)'
    r'\s*(?:\[[^\]\n]{0,40}\])?\s*[:\]]'         # FATAL: / FATAL [gpu]: / [FATAL]
    r'|traceback \(most recent call last\)'
    r'|segmentation fault|abort trap|bus error|core dumped'
    r'|terminate called|what\(\):|std::bad_alloc|std::runtime_error'
    r'|uncaught exception|assertion failed'
    r'|simulation failed',
    re.IGNORECASE)

_WARN_RE = re.compile(
    r'(?<![A-Za-z])warn(?:ing)?\b'               # "WARNING" / "warn:"
    r'|conservative defaults|no usable gpu profile'
    r'|falling back|fall back to|fallback|deprecat'
    r'|spice error|spice\(',                     # engine SPICE diagnostics
    re.IGNORECASE)

_PROGRESS_RE = re.compile(r'(\d+(?:\.\d+)?)\s*%')


def classify_severity(text):
    """Classify one output line as ``'error'``, ``'warning'`` or ``'info'``
    from its content, for stdout and stderr alike."""
    if _ERROR_RE.search(text):
        return "error"
    if _WARN_RE.search(text):
        return "warning"
    return "info"


def extract_progress(text):
    """Return the percentage in ``text`` as a float, or ``None``."""
    match = _PROGRESS_RE.search(text)
    return float(match.group(1)) if match else None


def sample_cputime(pid):
    """Accumulated CPU seconds of ``pid`` via ``ps``, or ``None``.

    Portable across macOS and Linux; ``ps`` prints ``[[dd-]hh:]mm:ss[.ff]``.
    """
    try:
        out = subprocess.run(
            ["ps", "-o", "cputime=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except Exception:
        return None
    if not out:
        return None
    try:
        days = 0
        if "-" in out:
            d, out = out.split("-", 1)
            days = int(d)
        parts = [float(p) for p in out.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, m, s = parts[-3], parts[-2], parts[-1]
        return days * 86400.0 + h * 3600.0 + m * 60.0 + s
    except (ValueError, IndexError):
        return None


# --- what this engine binary can be told on the command line ------------------
# The engine's flag set grows: `--thermal-inertia` and `--albedo` are being
# added, and a caller that wants to use them must find out whether the binary in
# front of it has them rather than assume. `--help` is the only interface that
# answers, so it is parsed -- once per binary, keyed on the binary's identity so
# a rebuild invalidates the answer.

_FLAG_CACHE = {}
_FLAG_TOKEN = re.compile(r"--[A-Za-z0-9][-A-Za-z0-9_]*")


def engine_flags(executable, timeout=20.0):
    """The set of ``--flags`` ``executable --help`` advertises.

    Returns an empty set if the binary cannot be run or prints no flags; that
    is reported on stderr rather than passed off as "supports nothing", because
    the two are different and only one of them is a caller's fault.
    """
    path = str(executable)
    try:
        stat = os.stat(path)
        key = (path, stat.st_mtime_ns, stat.st_size)
    except OSError as exc:
        import sys
        sys.stderr.write(f"[ptm] cannot stat engine {path!r}: {exc}\n")
        return frozenset()
    if key in _FLAG_CACHE:
        return _FLAG_CACHE[key]

    try:
        proc = subprocess.run([path, "--help"], capture_output=True, text=True,
                              timeout=timeout)
        text = (proc.stdout or "") + (proc.stderr or "")
    except (OSError, subprocess.SubprocessError) as exc:
        import sys
        sys.stderr.write(
            f"[ptm] could not ask {path!r} for its flags ({exc}); assuming it "
            f"advertises none\n")
        text = ""
    flags = frozenset(_FLAG_TOKEN.findall(text))
    if not flags and text:
        import sys
        sys.stderr.write(
            f"[ptm] {path} --help printed no --flags; assuming it advertises "
            f"none\n")
    _FLAG_CACHE[key] = flags
    return flags


def engine_supports_flag(executable, flag):
    """Whether ``executable`` advertises ``flag`` (e.g. ``'--albedo'``).

    Matched as a whole token, which is the entire point: today's engine
    advertises ``--albedo-map`` and ``--albedo-maps-directory`` but no
    ``--albedo``, and a substring test says yes to all three.
    """
    return flag in engine_flags(executable)


def build_command(executable, config_file, config_dict=None):
    """Build the engine argv for ``config_file``.

    Radiation settings are deliberately not duplicated onto the command line:
    ``config_parser.cpp`` reads the whole ``radiation:`` section from the YAML,
    and a CLI copy would silently override it whenever the two sets of
    defaults drifted apart. The two flags below are the ones the YAML parser
    does not cover.
    """
    command = [str(executable), "--config", str(config_file)]
    cfg = config_dict or {}
    if cfg.get("thermal", {}).get("use_gpu", False):
        command.append("--use-gpu-thermal")
    generic_cfg = cfg.get("generic_body", {}).get("config_file", "")
    if generic_cfg:
        command.extend(["--generic-body-config", str(generic_cfg)])
    return command


#: A run at or above this many simulated days is "long" for the purposes of the
#: GPU-wedge advisory below. 30 days is one polar-site production window and the
#: length at which the 2026-08 wedges occurred; shorter runs finish inside the
#: driver's healthy window even on the affected hardware.
LONG_RUN_DAYS = 30.0

#: Same, in output timesteps, for a config whose duration is short but whose
#: cadence is fine enough to keep the GPU busy for hours.
LONG_RUN_TIMESTEPS = 2000


def long_run_advisory(config_dict, chunked=False):
    """Messages to show before launching a LONG GPU run as a single process.

    The GUI, the ``ptm run`` CLI and the notebook helper all launch one
    ``ptm_thermal_sim`` directly. That is the right thing for a short run, and
    it means those users get none of the chunked runner's GPU-wedge guardrails:
    the inter-chunk cool-down, the heartbeat stall abort, and per-attempt logs
    that are rotated rather than overwritten. On macOS+AMD an overnight GPU run
    launched this way can wedge the machine and leave no record of where.

    Advisory only -- it never refuses and never reroutes. Rerouting a GUI run
    into the chunked workflow would change its output layout (merged
    statistics, no per-step series), which is a decision for the user, not for
    a warning. Returns a list of strings; empty when the advice does not apply.
    """
    if chunked:
        return []
    cfg = config_dict or {}
    uses_gpu = any(bool((cfg.get(s) or {}).get("use_gpu", False))
                   for s in ("thermal", "radiation", "illumination"))
    if not uses_gpu:
        return []
    sim = cfg.get("simulation") or {}
    try:
        days = float(sim.get("duration_days", 0.0) or 0.0)
    except (TypeError, ValueError):
        days = 0.0
    try:
        steps = int(sim.get("timesteps", 0) or 0)
    except (TypeError, ValueError):
        steps = 0
    if days < LONG_RUN_DAYS and steps < LONG_RUN_TIMESTEPS:
        return []
    return [
        f"[gpu-guardrails] this is a long GPU run ({days:g} simulated days, "
        f"{steps} timesteps) launched as ONE engine process. The chunked "
        f"runner adds the GPU-wedge guardrails this path does not have: an "
        f"inter-chunk cool-down, a stall abort that kills a wedged engine "
        f"instead of waiting for it, and per-attempt logs rotated to "
        f"logs.prev-<UTC> so a rerun cannot destroy the failed attempt's "
        f"record.",
        "[gpu-guardrails] to use it:  python3 scripts/run_chunked_simulation.py "
        "--config <this config> --total-days <days> --output-dir <dir>   "
        "(see docs/dev/gpu-wedge-guardrails.md). Note it produces merged "
        "statistics and no per-step series.",
    ]


class EngineProcess:
    """Run an engine command, reporting output through plain callables.

    ``on_output`` / ``on_warning`` / ``on_error`` each take one string;
    ``on_progress`` takes a float percentage; ``on_heartbeat`` takes
    ``(message, stalled)`` and fires every ``WATCHDOG_INTERVAL_S`` while the
    child's CPU clock is not advancing, and once when it resumes. Any of them
    may be ``None``.

    ``timeout`` is a wall-clock ceiling in seconds. On expiry the child is
    stopped through :meth:`stop` and :attr:`timed_out` is set, so a caller can
    tell "killed at the ceiling" from "exited non-zero on its own" -- an
    unattended sweep needs that distinction and an exit status alone does not
    carry it.

    ``log_path`` tees every line the child writes to a file, in addition to the
    callbacks, which are untouched. A GUI-launched run used to leave NO engine
    stdout or stderr anywhere on disk: after the 2026-09-02 site A flux-cube run
    the ``[PTM_HMATRIX_VF_REPLACE] ROUTE = ...`` banner and the whole
    coupling-convergence report were simply gone, and which view-factor matrix
    had been used had to be reconstructed from the product's own header. The
    console is not a record -- it is a widget that is cleared on the next
    launch. Default ``None`` keeps every existing caller byte-identical.

    :meth:`run` blocks and returns the child's exit status.
    """

    WATCHDOG_INTERVAL_S = 30.0   # child cputime sampling cadence
    STALL_CPU_DELTA_S = 0.5      # less CPU advance than this per interval = stalled

    def __init__(self, command, working_dir=None, env=None,
                 on_output=None, on_warning=None, on_error=None,
                 on_progress=None, on_heartbeat=None, timeout=None,
                 log_path=None):
        self.command = [str(c) for c in command]
        self.working_dir = str(working_dir) if working_dir else None
        self.env = env
        self.on_output = on_output
        self.on_warning = on_warning
        self.on_error = on_error
        self.on_progress = on_progress
        self.on_heartbeat = on_heartbeat
        self.timeout = None if timeout is None else float(timeout)
        self.timed_out = False
        self.process = None
        self.returncode = None
        self.log_path = str(log_path) if log_path else None
        self._log = None
        self._stop_requested = False

    # -- the on-disk record -------------------------------------------------
    def _open_log(self):
        """Open ``log_path`` for appending, or say why there will be no log.

        APPEND, never truncate: a chunked or resumed run writes into the same
        output directory more than once, and a second launch that erased the
        first one's diagnosis would recreate the gap this exists to close.
        Line-buffered so the file is complete up to the last line the engine
        managed to write -- which is the case that matters, because a crash is
        exactly when the record is wanted and exactly when nothing gets
        flushed on the way out.

        A log that cannot be opened is reported through ``on_warning`` and the
        run continues. Failing the run over its own diagnostics would be worse
        than the gap; being silent about it would be the same defect one level
        down.
        """
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._log = open(self.log_path, "a", buffering=1, encoding="utf-8",
                             errors="replace")
        except OSError as exc:
            self._log = None
            self._emit(self.on_warning,
                       f"WARNING: no engine log will be written to "
                       f"{self.log_path} ({exc}); the console is the only "
                       f"record of this run")
            return
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self._write_log(f"===== engine start {stamp} =====")
        self._write_log("command: " + " ".join(shlex.quote(c) for c in self.command))
        self._write_log(f"cwd: {self.working_dir}")
        self._emit(self.on_output, f"[run] engine log: {self.log_path}")

    def _write_log(self, text):
        if self._log is None:
            return
        try:
            self._log.write(text + "\n")
        except (OSError, ValueError):
            # ValueError = already closed. A trace that stops the run it is
            # tracing would be worse than no trace.
            self._log = None

    def _close_log(self, returncode):
        if self._log is None:
            return
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        self._write_log(f"===== engine exit status {returncode} at {stamp} "
                        f"(timed_out={self.timed_out}) =====")
        try:
            self._log.close()
        except OSError:
            pass
        self._log = None

    # -- callback plumbing --------------------------------------------------
    def _emit(self, cb, *args):
        if cb is not None:
            cb(*args)

    def _emit_classified(self, text):
        # The file first: a callback that raises must not cost the record.
        self._write_log(text)
        severity = classify_severity(text)
        if severity == "error":
            self._emit(self.on_error, text)
        elif severity == "warning":
            self._emit(self.on_warning, text)
        else:
            self._emit(self.on_output, text)

    def _drain(self, buf, final=False):
        """Emit each complete line in ``buf``; keep the unterminated tail back.

        ``final=True`` emits that trailing segment as well, for the one call
        made after both pipes are at EOF: nothing more is coming, so a last
        line with no newline on it is complete after all. Everything the child
        wrote goes through here, so a line delivered late is still split,
        still classified, and still counted as progress.
        """
        parts = re.split(b'[\r\n]', buf)
        if final:
            complete, remainder = parts, b""
        else:
            complete, remainder = parts[:-1], (parts[-1] if parts else b"")
        for seg in complete:
            if not seg:
                continue
            decoded = seg.decode('utf-8', errors='replace').strip()
            if not decoded:
                continue
            progress = extract_progress(decoded)
            if progress is not None:
                self._emit(self.on_progress, progress)
            self._emit_classified(decoded)
        buf.clear()
        buf.extend(remainder)

    # -- execution ----------------------------------------------------------
    def run(self):
        """Start the child, pump both pipes to exhaustion, return its status."""
        # Before Popen, so a child that fails to start at all still leaves the
        # command and the working directory on disk to look at.
        self._open_log()
        try:
            return self._run_inner()
        finally:
            # `finally`, not the end of the happy path: an exception on the way
            # out is precisely when the record is wanted, and the caller sees
            # nothing else.
            self._close_log(self.returncode)

    def _run_inner(self):
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=False,
            bufsize=0,
            cwd=self.working_dir,
            env=self.env,
            start_new_session=True   # own process group: clean group kill
        )

        stdout_buffer = bytearray()
        stderr_buffer = bytearray()

        started = time.monotonic()
        next_watchdog = started + self.WATCHDOG_INTERVAL_S
        last_cputime = None
        stall_since = None

        while not self._stop_requested:
            if (self.timeout is not None
                    and (time.monotonic() - started) > self.timeout
                    and self.process.poll() is None):
                self.timed_out = True
                self._emit_classified(
                    f"ERROR: engine exceeded the {self.timeout:g} s wall-clock "
                    f"timeout; stopping it")
                self.stop()
                break
            if self.process.poll() is not None:
                tail_out = self.process.stdout.read() or b""
                tail_err = self.process.stderr.read() or b""
                if tail_out:
                    stdout_buffer.extend(tail_out)
                if tail_err:
                    stderr_buffer.extend(tail_err)
                break

            now = time.monotonic()
            if now >= next_watchdog:
                next_watchdog = now + self.WATCHDOG_INTERVAL_S
                cpu = sample_cputime(self.process.pid)
                if cpu is not None:
                    if (last_cputime is not None
                            and (cpu - last_cputime) < self.STALL_CPU_DELTA_S):
                        if stall_since is None:
                            stall_since = now - self.WATCHDOG_INTERVAL_S
                        mins = (now - stall_since) / 60.0
                        msg = (f"No progress for {mins:.1f} min (CPU idle) -- "
                               "likely blocked on file I/O (cloud-synced or "
                               "dataless file?)")
                        # In the log too: "did it wedge, and when" is the
                        # question this file gets asked most, and the answer
                        # is not in the engine's own output.
                        self._write_log(f"[watchdog] {msg}")
                        self._emit(self.on_heartbeat, msg, True)
                    else:
                        if stall_since is not None:
                            self._write_log("[watchdog] Progress resumed (CPU active)")
                            self._emit(self.on_heartbeat,
                                       "Progress resumed (CPU active)", False)
                        stall_since = None
                    last_cputime = cpu

            readable, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 0.1
            )
            for stream in readable:
                data = stream.read(4096)
                if not data:
                    continue
                if stream is self.process.stdout:
                    stdout_buffer.extend(data)
                    self._drain(stdout_buffer)
                else:
                    stderr_buffer.extend(data)
                    self._drain(stderr_buffer)

        # Whatever is left, through the same splitter as everything else. The
        # exit branch above reads both pipes to EOF in one go, so this last
        # chunk routinely holds the whole end of the run -- every line the
        # child wrote after the reader's previous pass. Emitting it unsplit
        # welded those lines into one: no on_progress for any of them, and a
        # consumer that filters or throttles on the first percentage in a
        # "line" then acted on all of them at once.
        for buf in (stdout_buffer, stderr_buffer):
            self._drain(buf, final=True)

        self.process.wait()
        self.returncode = self.process.returncode
        return self.returncode

    def stop(self):
        """Stop the child: TERM to its process group, KILL after 5 s.

        A process stuck in an uninterruptible cloud-file read may not die
        immediately even on KILL, so both waits are bounded and this never
        hangs the caller.
        """
        self._stop_requested = True
        if self.process and self.process.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass   # bounded: never hang on a stuck process

    def _signal_group(self, sig):
        """Signal the child's process group; fall back to the child alone."""
        try:
            os.killpg(os.getpgid(self.process.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                if sig == signal.SIGKILL:
                    self.process.kill()
                else:
                    self.process.terminate()
            except Exception:
                pass


# --- run it to completion -----------------------------------------------------
# EngineProcess is callback-shaped because the GUI needs it to be: a QThread
# forwards each callback to a signal while the event loop keeps running. A
# script or a notebook wants the opposite -- one blocking call that returns what
# happened -- and every consumer in the tree was writing the same 15-to-48-line
# adapter to get there. run_engine() is that adapter, once.
#
# It serves the two consumers that actually exist: one wants a throttled live
# log (echo=True), one wants only the text afterwards (echo=False, .text).

#: The engine's second progress form. It writes two carriage-return readouts:
#: a ``[====>] 47%`` bar (matched by :func:`extract_progress`) and this one,
#: which carries no percent sign and so needs its own pattern.
_STEP_RE = re.compile(r'Progress (\d+)/(\d+)')

_BUILD_HINT = (
    "Build it from the repository root (INSTALL.md, 'TL;DR'):\n"
    "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTM_ENABLE_GPU=OFF \\\n"
    "          -DPTM_ARCH_NATIVE=OFF -DCSPICE_ROOT=/path/to/cspice\n"
    "    cmake --build build --target ptm_thermal_sim -j\n"
    "or point $PTM_BUILD_DIR at an existing build/bin."
)


class CompletedRun:
    """What one engine run did. Returned by :func:`run_engine`.

    ``lines`` are the engine's output lines in the order they were read, both
    streams interleaved and carriage-return segments split apart, which is how
    the engine's progress bars arrive. ``text`` joins them.

    ``output_dir`` is where the run wrote, taken from the config's
    ``output.directory`` (resolved against ``cwd``) or from an ``--output``
    override in ``extra_args``. It is ``None`` when neither says, and
    :meth:`results` then raises rather than guessing.
    """

    __slots__ = ("command", "returncode", "lines", "seconds", "timed_out",
                 "config", "output_dir", "executable")

    def __init__(self, command, returncode, lines, seconds, timed_out=False,
                 config=None, output_dir=None, executable=None):
        self.command = list(command)
        self.returncode = returncode
        self.lines = list(lines)
        self.seconds = float(seconds)
        self.timed_out = bool(timed_out)
        self.config = config
        self.output_dir = output_dir
        self.executable = executable

    def __repr__(self):
        return (f"<CompletedRun rc={self.returncode} "
                f"{len(self.lines)} lines {self.seconds:.1f}s "
                f"-> {self.output_dir}>")

    @property
    def ok(self):
        """True when the engine exited 0 and was not stopped at a timeout."""
        return self.returncode == 0 and not self.timed_out

    @property
    def text(self):
        """The whole output as one string."""
        return "\n".join(self.lines)

    def matching(self, *needles):
        """Output lines containing any of ``needles`` -- the log-grep every
        caller writes after a quiet run."""
        return [ln for ln in self.lines
                if any(n in ln for n in needles)]

    def results(self, **kwargs):
        """A :class:`ptm.results.Results` for :attr:`output_dir`.

        ``**kwargs`` go to the constructor (``body_radius_km=``,
        ``units_style=``). Raises if the run did not say where it wrote, or
        wrote nothing there.
        """
        if self.output_dir is None:
            raise ValueError(
                "this run's output directory is unknown: the config set no "
                "output.directory and no --output was passed. Construct "
                "Results(<path>) yourself.")
        if not Path(self.output_dir).is_dir():
            raise FileNotFoundError(
                f"the engine exited {self.returncode} but {self.output_dir} "
                f"does not exist; the run wrote nothing there")
        from .results import Results
        return Results(self.output_dir, **kwargs)


class _Throttle:
    """Decide which of the engine's progress readouts to echo.

    The engine emits a percent roughly 100 times per phase and a step counter
    once per output step. Echoing all of them turns an executed notebook into
    several hundred lines of bar art. This keeps the first, the last, and one
    per ``every`` percent of advance, per readout kind.

    A percentage that goes *backwards* means a new phase started -- the engine
    runs an illumination bar and then a thermal bar, both 0 to 100 -- so the
    counter resets. Without that the second bar is silent until it finishes,
    which is exactly the phase a reader is waiting on.
    """

    def __init__(self, every):
        self.every = float(every)
        self.last = {}

    def keep(self, kind, pct):
        if self.every <= 0:
            return True
        previous = self.last.get(kind)
        if previous is not None and pct < previous:
            previous = None                      # new phase
        if pct >= 100.0 or previous is None or pct - previous >= self.every:
            self.last[kind] = pct
            return True
        return False


def _resolve_output_dir(config_dict, cwd, extra_args):
    """Where the engine will write, or ``None`` if nothing says."""
    from .engine_config import get_dotted
    directory = None
    args = [str(a) for a in extra_args]
    for i, arg in enumerate(args):
        if arg == "--output" and i + 1 < len(args):
            directory = args[i + 1]
        elif arg.startswith("--output="):
            directory = arg.split("=", 1)[1]
    if directory is None:
        directory = get_dotted(config_dict or {}, "output.directory")
    if not directory:
        return None
    path = Path(directory).expanduser()
    return path if path.is_absolute() else (Path(cwd) / path).resolve()


def _load_config(config):
    """The config as a dict, or ``{}`` with a warning if it cannot be read.

    Warned rather than raised: the engine is the authority on whether a config
    is usable, and refusing to launch because *this* parser choked would make
    the wrapper stricter than the thing it wraps.
    """
    try:
        import yaml
    except ImportError:                                        # pragma: no cover
        sys.stderr.write("[ptm] PyYAML is not installed; run_engine cannot read "
                         "the config for --use-gpu-thermal / output.directory\n")
        return {}
    try:
        loaded = yaml.safe_load(Path(config).read_text())
    except Exception as exc:                                   # noqa: BLE001
        sys.stderr.write(
            f"[ptm] could not read {config} as YAML ({exc}); launching anyway "
            f"and letting the engine judge it. output_dir will be unknown.\n")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def resolve_engine(exe=None, cwd=None, config=None, name="ptm_thermal_sim"):
    """The engine executable to run, and the working directory to run it in.

    Returns ``(executable, cwd)`` as absolute :class:`~pathlib.Path` objects.
    ``cwd`` defaults to the checkout containing ``config`` (falling back to the
    one containing the current directory), because the engine resolves
    ``output.directory`` and the other relative paths in a config against the
    process working directory, and every shipped config is written to be run
    from the repository root.

    Raises :class:`FileNotFoundError` naming the build command when no engine
    is found, which is the failure every notebook hand-writes a message for.
    """
    root = None
    if cwd is None:
        if config is not None:
            root = paths.find_repo_root(Path(config).expanduser().parent)
        if root is None:
            root = paths.find_repo_root()
        cwd = root if root is not None else Path.cwd()
    cwd = Path(cwd).expanduser().resolve()

    if exe is None:
        search = [cwd]
        if root is not None and root != cwd:
            search.append(root)
        exe = paths.find_engine_executable(name, search_roots=search)
        if exe is None:
            raise FileNotFoundError(
                f"{name} not found in {cwd}/build/bin (nor in $PTM_BUILD_DIR). "
                f"{_BUILD_HINT}")
    exe = Path(exe).expanduser().resolve()
    if not exe.exists():
        raise FileNotFoundError(f"{exe} does not exist. {_BUILD_HINT}")
    return exe, cwd


# --- which binary am I about to run? ------------------------------------------
# The engine now records its own identity into every run directory, and
# Results.provenance() reads it back -- md5, git SHA, version, build type, the
# command line. That answers "what produced this run" completely.
#
# It cannot answer "what am I about to run", which is the question a notebook
# asks in its first cell and a script asks before a twenty-minute launch. There
# is no run directory yet.


class BinaryProvenance:
    """The identity of an engine binary, before it has produced anything.

    The run-directory counterpart is
    :meth:`ptm.results.Results.provenance`, which reads what the engine wrote
    about itself; use that one to describe a finished run. This one describes a
    binary on disk.

    ``print(...)`` gives the four lines a caller wants in a log header. Fields
    that could not be determined are ``None`` and are listed in
    :attr:`unavailable` rather than left to look like absences.
    """

    __slots__ = ("path", "md5", "size_bytes", "mtime_utc", "repo_root",
                 "git_sha", "git_branch", "git_dirty", "help_ok", "n_flags",
                 "unavailable")

    def __init__(self, **fields):
        unknown = set(fields) - set(self.__slots__)
        if unknown:
            # A typo'd keyword would otherwise land nowhere and leave the field
            # it was meant for reading None, i.e. "not determined".
            raise TypeError(f"unknown BinaryProvenance field(s): {sorted(unknown)}")
        for name in self.__slots__:
            setattr(self, name, fields.get(name))
        self.unavailable = tuple(fields.get("unavailable", ()))

    def __repr__(self):
        return f"<BinaryProvenance {self.path} md5={self.md5}>"

    def __str__(self):
        lines = [f"Binary : {self.path}",
                 f"  md5  : {self.md5}",
                 f"  built: {self.mtime_utc}  ({self.size_bytes} bytes)"]
        if self.git_sha:
            dirty = " +uncommitted changes" if self.git_dirty else ""
            lines.append(f"  commit: {self.git_sha} ({self.git_branch}){dirty}")
        if self.help_ok is not None:
            lines.append(f"  --help: {'ok' if self.help_ok else 'FAILED'}"
                         f"  ({self.n_flags} flags advertised)")
        for item in self.unavailable:
            lines.append(f"  (not determined: {item})")
        return "\n".join(lines)

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


def file_md5(path, block=1 << 20):
    """md5 of a file, read in blocks.

    Block-wise rather than ``read_bytes()``: the engine binary is a few
    megabytes and a view-factor matrix can be gigabytes, and nothing here needs
    the contents afterwards. ``ptm.sweep`` uses this for the same purpose.
    """
    import hashlib
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root, *args):
    """Run git in ``repo_root``; return stdout, or ``None``."""
    import shutil
    git = shutil.which("git")
    if git is None or repo_root is None:
        return None
    try:
        proc = subprocess.run([git, *args], cwd=str(repo_root),
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def binary_provenance(exe=None, repo_root=None, check_help=True):
    """Identify the engine binary a run would use.

    ::

        from ptm import run as engine
        print(engine.binary_provenance())

        Binary : /path/to/build/bin/ptm_thermal_sim
          md5  : e19adfdabfac617d187833414acc7931
          built: 2026-08-10T02:36:11Z  (3287688 bytes)
          commit: 3f06348 (feat/ptm-api-gaps)
          --help: ok  (126 flags advertised)

    ``exe`` defaults to the engine :func:`resolve_engine` would pick.
    ``check_help`` runs ``--help`` once, which is the cheapest check that a
    build is not broken; the flag count comes from the same call, cached.

    Git is located with ``shutil.which``, not hard-coded, and a checkout with no
    git available reports that in :attr:`BinaryProvenance.unavailable` instead
    of silently omitting the commit.
    """
    from datetime import datetime, timezone

    exe, cwd = resolve_engine(exe=exe, cwd=repo_root)
    if repo_root is None:
        repo_root = paths.find_repo_root(exe.parent) or cwd

    unavailable = []
    stat = exe.stat()
    md5 = file_md5(exe)
    mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)

    sha = _git(repo_root, "rev-parse", "--short", "HEAD")
    branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = None
    if sha is None:
        unavailable.append(
            f"git commit ({repo_root} is not a git checkout, or git is not "
            f"on PATH)")
    else:
        status = _git(repo_root, "status", "--porcelain")
        dirty = bool(status)

    help_ok, n_flags = None, None
    if check_help:
        flags = engine_flags(exe)
        n_flags = len(flags)
        help_ok = n_flags > 0
        if not help_ok:
            unavailable.append(
                f"the flag list: {exe} --help advertised none, which usually "
                f"means the build is broken")

    return BinaryProvenance(
        path=exe, md5=md5, size_bytes=stat.st_size,
        mtime_utc=mtime.strftime("%Y-%m-%dT%H:%M:%SZ"), repo_root=repo_root,
        git_sha=sha, git_branch=branch, git_dirty=dirty,
        help_ok=help_ok, n_flags=n_flags, unavailable=tuple(unavailable))


def run_engine(config, exe=None, cwd=None, echo=False, progress_every=25.0,
               extra_args=(), env=None, timeout=None, show=(), check=True):
    """Run ``ptm_thermal_sim`` on ``config`` to completion.

    ::

        from ptm import run as engine
        done = engine.run_engine("examples/quickstart_moon_diurnal.yaml",
                                 echo=True)
        result = done.results()                 # a ptm.results.Results

    ``exe`` and ``cwd`` default through :func:`resolve_engine`. ``extra_args``
    are appended to the command line, so ``--equilibration-orbits 5`` and
    ``--output some/dir`` work and the second one is reflected in
    :attr:`CompletedRun.output_dir`.

    ``echo=False`` (the default) runs silently and keeps everything in
    :attr:`CompletedRun.lines`; ``show=('Simulation completed in',)`` prints
    just the lines you care about once it is over. ``echo=True`` streams the
    output live, keeping one progress readout per ``progress_every`` percent so
    an executed notebook stays readable; ``progress_every=0`` echoes every
    line, bar art included.

    ``check=True`` raises :class:`RuntimeError` on a non-zero exit, quoting the
    last 30 lines -- the engine's own FATAL message is in there. Pass
    ``check=False`` to inspect :attr:`CompletedRun.returncode` yourself.

    No physics: this starts a process, reads what it printed, and reports it.
    """
    config = Path(config).expanduser()
    exe, cwd = resolve_engine(exe=exe, cwd=cwd, config=config)
    config_dict = _load_config(config)
    command = build_command(exe, config, config_dict) + [str(a) for a in extra_args]
    output_dir = _resolve_output_dir(config_dict, cwd, extra_args)

    lines = []
    throttle = _Throttle(progress_every)

    def take(line):
        lines.append(line)
        if not echo:
            return
        pct = extract_progress(line)
        step = _STEP_RE.search(line)
        if pct is not None:
            if throttle.keep("percent", pct):
                print(line)
        elif step is not None:
            done, total = int(step.group(1)), int(step.group(2))
            if throttle.keep("step", 100.0 * done / max(total, 1)):
                print(line)
        else:
            print(line)

    if echo:
        print("Running:", " ".join(str(c) for c in command))
        print(f"(cwd={cwd})\n")
    for msg in long_run_advisory(config_dict):
        print(msg, file=sys.stderr)

    started = time.time()
    process = EngineProcess(command, working_dir=cwd, env=env, on_output=take,
                            on_warning=take, on_error=take, timeout=timeout)
    returncode = process.run()
    seconds = time.time() - started

    completed = CompletedRun(command, returncode, lines, seconds,
                             timed_out=process.timed_out, config=config,
                             output_dir=output_dir, executable=exe)

    for line in completed.matching(*show):
        print("   ", line.strip())
    if echo:
        print(f"\n--- {len(lines)} output lines; {seconds:.1f} s wall; "
              f"exit code {returncode} ---")

    if output_dir is not None and returncode == 0 and not Path(output_dir).is_dir():
        sys.stderr.write(
            f"[ptm] the engine exited 0 but {output_dir} was not created; the "
            f"config's output.directory may not be what this run used\n")
    if check and returncode != 0:
        raise RuntimeError(
            f"{config} exited with code {returncode}"
            f"{' (stopped at the timeout)' if process.timed_out else ''}:\n"
            + "\n".join(lines[-30:]))
    return completed
