"""Launching ``ptm_thermal_sim`` and reading what it says while it runs.

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

import os
import re
import select
import signal
import subprocess
import time

from . import paths

__all__ = [
    "EngineProcess",
    "build_command",
    "classify_severity",
    "engine_flags",
    "engine_supports_flag",
    "extract_progress",
    "find_engine_executable",
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

    :meth:`run` blocks and returns the child's exit status.
    """

    WATCHDOG_INTERVAL_S = 30.0   # child cputime sampling cadence
    STALL_CPU_DELTA_S = 0.5      # less CPU advance than this per interval = stalled

    def __init__(self, command, working_dir=None, env=None,
                 on_output=None, on_warning=None, on_error=None,
                 on_progress=None, on_heartbeat=None, timeout=None):
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
        self._stop_requested = False

    # -- callback plumbing --------------------------------------------------
    def _emit(self, cb, *args):
        if cb is not None:
            cb(*args)

    def _emit_classified(self, text):
        severity = classify_severity(text)
        if severity == "error":
            self._emit(self.on_error, text)
        elif severity == "warning":
            self._emit(self.on_warning, text)
        else:
            self._emit(self.on_output, text)

    def _drain(self, buf):
        parts = re.split(b'[\r\n]', buf)
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
                        self._emit(self.on_heartbeat,
                                   f"No progress for {mins:.1f} min (CPU idle) -- "
                                   "likely blocked on file I/O (cloud-synced or "
                                   "dataless file?)", True)
                    else:
                        if stall_since is not None:
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

        # Whatever is left, classified, so a genuine error in the final chunk
        # is still reported as one.
        for buf in (stdout_buffer, stderr_buffer):
            if buf:
                try:
                    tail = buf.decode('utf-8', errors='replace').strip()
                    if tail:
                        self._emit_classified(tail)
                except Exception:
                    pass

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
