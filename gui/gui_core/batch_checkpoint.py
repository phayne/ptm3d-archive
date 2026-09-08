"""Banking completed facet batches so an interrupted run resumes: the rules,
with no Qt.

WHAT THE FEATURE IS. The thermal solver works through the mesh one small group
of facets at a time, and it carries each group through the whole time span
before it starts the next one. So there is no moment during a run at which the
whole mesh has reached the same step -- but there IS a moment at which a group
of facets is finished for the entire run. ``thermal.batch_checkpoint`` banks
those finished groups to disk. If the run is interrupted, starting it again
reads them back and integrates only the groups that were never finished.

WHY THERE IS NO "SAVE EVERY N MINUTES" CONTROL. A control like that would be a
promise this model cannot keep. Saving the whole mesh partway through would
save facets that have not been solved yet, which is worse than saving nothing,
because the file would look complete. The unit that can honestly be saved is a
finished facet group, so that is the unit this offers.

WHAT IT BOUNDS, AND WHAT IT DOES NOT. A run interrupted inside a facet group
loses that group's whole time integration -- the groups are small, so this is a
small loss, but it is not zero. Bounding the loss more tightly than that is the
job of splitting a long run into shorter runs, which the command-line runner
already does and does exactly. This is stated on screen rather than left to be
discovered: a safety feature that quietly bounds less than a reader assumes is
worse than none.

THE SIZE IS THE THING TO LOOK AT. What gets banked is dominated by the surface
temperature at every step, for every facet: eight bytes per facet per step. On
a fine mesh over a long window that is tens of gigabytes, not the megabytes a
reader might expect from the phrase "checkpoint". ``batch_checkpoint_max_gib``
refuses above a ceiling for that reason, and the figure is put on screen beside
the switch.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.

* **It does not decide how many steps a run will integrate.** The banked size
  counts the spin-up as well as the reported window, and how many spin-up steps
  a run takes is settled by the solver from the equilibration settings and the
  body's rotation. Reproducing that here would be a second copy of a rule that
  already exists once, and a second copy that disagreed would be worse than no
  figure. So the size reported here is a LOWER BOUND, labeled as one, over the
  steps that are known from the configuration; the model prints the exact
  figure in its first seconds and stops there if it is over the ceiling.

* **It does not delete anything.** Banked work is work, and a button that
  throws it away is a button that eventually throws away the wrong run's. When
  banked groups are found, the launch check names the directory holding them so
  it can be cleared deliberately.

* **It does not decide whether the groups on disk belong to this run.** The
  model re-reads and re-verifies every banked file against the mesh, the depth
  grid, the epoch, the window and the forcing before it trusts one, and refuses
  the whole resume if any of them disagrees. Guessing that here would put a
  second, weaker opinion beside a strict one.

THE INDEX IS NOT THE AUTHORITY. The directory holds a ``manifest.json`` listing
what has been banked, and the model treats it as an index only: after an
interruption the last file can be on disk while the index has not caught up.
Anything this module reports from the index is reported as what the index says,
and the count of files on disk is reported beside it, so the two can be seen to
disagree.
"""

import json
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Keys
# ---------------------------------------------------------------------------

#: The four keys the model reads, spelled out rather than assembled: a key
#: built by string surgery is a key no search finds.
ENABLE_KEY = "thermal.batch_checkpoint"
DIR_KEY = "thermal.batch_checkpoint_dir"
MAX_GIB_KEY = "thermal.batch_checkpoint_max_gib"
FLUSH_MIB_KEY = "thermal.batch_checkpoint_flush_mib"

#: Their leaf names, for reading a ``thermal:`` section directly.
ENABLE_LEAF = "batch_checkpoint"
DIR_LEAF = "batch_checkpoint_dir"
MAX_GIB_LEAF = "batch_checkpoint_max_gib"
FLUSH_MIB_LEAF = "batch_checkpoint_flush_mib"

# ---------------------------------------------------------------------------
# 2. The file layout, transcribed from the model
# ---------------------------------------------------------------------------

#: Where the banked groups go when ``batch_checkpoint_dir`` is left empty:
#: below the run's own output directory (``ptm_thermal_sim.cpp``, where the
#: switch and the directory are resolved together).
DEFAULT_SUBDIR = "_thermal_state/partial"

#: One banked file is named ``part_gNNNN_sMMMMMM.ptmb``; the model recognizes
#: them by the suffix alone, and so does this.
PARTIAL_SUFFIX = ".ptmb"

#: The index beside them.
MANIFEST_NAME = "manifest.json"

#: Bytes banked per facet, per section of a banked file
#: (``batch_checkpoint.h``, ``bytesPerFace``). Named separately because they
#: are switched on separately: the surface series always, the final column only
#: when a final state file is being written, the depth accumulators only when
#: subsurface maps are switched on.
SURFACE_BYTES_PER_STEP = 8
COLUMN_BYTES_PER_NODE = 4
DEPTH_BYTES_PER_DEPTH = 24
DEPTH_BYTES_FIXED = 4

#: How many facets are in one group. The loss from an interruption is bounded
#: by one group's time integration, so the number belongs on screen beside the
#: claim (``thermal_model.h``, ``THERMAL_BATCH_SIZE``).
BATCH_FACES = 8

#: One gibibyte, so the ceiling and the estimate are compared in the units the
#: key is written in.
GIB = 1 << 30


def _section(config, name):
    node = (config or {}).get(name)
    return node if isinstance(node, dict) else {}


def _truthy(value, default=False):
    """A YAML scalar read as a boolean, tolerating the spellings a hand-edited
    file uses -- the same tolerance the rest of this package applies, so a
    malformed value never aborts a window."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("true", "yes", "on", "1")


def _number(value, default=0.0):
    """A YAML scalar read as a number, or `default`. Never raises: this module
    is reached from a Qt slot, where an escaping exception is not an error
    message but an exit."""
    if isinstance(value, bool) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _commas(n):
    return "{:,}".format(int(n))


def fmt_bytes(n):
    """A size in the units a person reads it in."""
    n = float(n)
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if n >= scale:
            return "%.1f %s" % (n / scale, unit)
    return "%d bytes" % int(n)


# ---------------------------------------------------------------------------
# 3. Reading the configuration
# ---------------------------------------------------------------------------

def enabled(config):
    """Is the feature switched on in this configuration?"""
    return _truthy(_section(config, "thermal").get(ENABLE_LEAF), False)


def ceiling_gib(config):
    """The configured ceiling in gibibytes. Zero or less means no ceiling --
    the model's own reading (``if (max_gib > 0.0)``), so a configuration that
    switches the ceiling off is described here the way the run will treat it."""
    return _number(_section(config, "thermal").get(MAX_GIB_LEAF), 64.0)


def partial_dir(config, output_directory=None):
    """Where the banked groups will go, as the model resolves it.

    An explicit ``batch_checkpoint_dir`` wins; otherwise it is below the run's
    output directory. `output_directory` overrides the configuration's own, for
    a caller that has already resolved it to an absolute path.
    """
    thermal = _section(config, "thermal")
    explicit = str(thermal.get(DIR_LEAF) or "").strip()
    if explicit:
        return explicit
    base = output_directory
    if base is None:
        base = str(_section(config, "output").get("directory") or "").strip()
    base = str(base or "").rstrip("/")
    if not base:
        return DEFAULT_SUBDIR
    return base + "/" + DEFAULT_SUBDIR


def final_state_conflict(config, output_directory=None):
    """The one refusal about paths that is decidable from the configuration.

    The model refuses when the finished state file sits inside the directory
    holding the banked groups: the sweep that finds banked files would see the
    finished state as a stray, and a cleanup would see it as a banked group.
    Returns the message, or an empty string.
    """
    final = str(_section(config, "thermal").get("final_state_file") or "").strip()
    if not final or not enabled(config):
        return ""
    directory = partial_dir(config, output_directory)
    if not directory:
        return ""
    if final.startswith(directory):
        return (
            "The file this run saves its finished state to (%s) is inside the "
            "directory the part-finished work goes to (%s). The model refuses "
            "that, because the sweep that looks for part-finished work would "
            "see the finished state as a stray file and a cleanup would see it "
            "as part-finished work. Point them at different places."
            % (final, directory))
    return ""


# ---------------------------------------------------------------------------
# 4. What the model will accept
# ---------------------------------------------------------------------------

@dataclass
class SupportPolicy:
    """Whether a configuration can bank finished facet groups, and why not if
    it cannot.

    ``blockers`` are conditions the model itself stops on; ``notes`` are true
    statements about a run that will succeed.
    """
    blockers: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def allowed(self):
        return not self.blockers

    @property
    def reason(self):
        return " ".join(self.blockers) if self.blockers else " ".join(self.notes)


def engine_support(config):
    """The model's own preconditions, transcribed.

    Each is a hard stop taken before any expensive work, so learning about one
    from the run is cheap. They are answered here anyway, because a launch that
    stops in its first seconds still interrupts the person who started it, and
    the reason is easier to act on beside the control that caused it.

    * **The thermal model runs on the graphics cards.** Banking a finished
      facet group is implemented on the processor solver only in this version.
      Proving a resume correct means interrupting a run abruptly, and doing
      that to a run with work queued on the graphics cards is the failure this
      workstation is known for, so that path was left for its own change.
    * **Facet-to-facet infrared exchange is on.** The exchange loop enters the
      whole-mesh solver more than once, and banked groups that survived from
      one entry to the next would let the second pass read back the first
      pass's results and call the run finished.
    * **The spin-up is set to end on its own convergence test.** That test can
      shorten the spin-up, so a resumed run would integrate a different length
      than the run it resumed from.

    ⛔ NOT A BLOCKER, and this was MEASURED rather than assumed: a run with
    ``thermal.run_thermal_mesh`` OFF and the switch ON finishes with status 0
    and does not mention the switch once. Nothing is banked and nothing says
    so. Reporting it as a blocker here would refuse a launch the model
    accepts, so it is reported by :func:`inert_warning` instead -- as the true
    statement that the setting will do nothing.

    NOT checked here, deliberately: the legacy scalar processor solver, which
    the model also refuses. It is selected by a command-line flag with no
    configuration key, and this application never passes that flag, so a check
    for it could not fire and would only look like coverage.
    """
    policy = SupportPolicy()
    thermal = _section(config, "thermal")
    radiation = _section(config, "radiation")

    if _truthy(thermal.get("use_gpu"), False):
        policy.blockers.append(
            "The thermal model is set to run on the graphics cards, and "
            "banking finished facet groups is implemented on the processor "
            "solver only in this version. The model stops rather than run with "
            "the setting quietly doing nothing. Run the thermal model on the "
            "processor, or leave this off.")

    if _truthy(radiation.get("enabled"), False):
        policy.blockers.append(
            "Facet-to-facet infrared exchange is on. The exchange loop enters "
            "the whole-mesh solver more than once, and work banked in one pass "
            "would be read back into the next -- which would report the first "
            "pass's temperatures as the finished run. The model stops rather "
            "than allow it. To bound an interruption on a run with the "
            "exchange on, split it into shorter runs instead.")

    if _truthy(thermal.get("equilibration_stop_on_certificate"), False):
        policy.blockers.append(
            "The spin-up is set to end when its own convergence test is met. "
            "That test can shorten the spin-up, so a run resumed from banked "
            "work would integrate a different length than the run that banked "
            "it. Choose one or the other.")

    if policy.allowed:
        policy.notes.append(
            "Banking finished facet groups changes no temperature anywhere: it "
            "writes the results a group has already reached, and reads them "
            "back instead of computing them again.")
    return policy


def inert_warning(config):
    """The case where the setting is on and will quietly do nothing.

    ⛔ MEASURED, on the model itself, not reasoned about: with
    ``thermal.run_thermal_mesh`` off and ``thermal.batch_checkpoint`` on the
    run completes with status 0 and the words "batch" and "checkpoint" appear
    nowhere in its output. There is no whole-mesh solve, so the switch is never
    reached, and every other unsupported combination stops the run and says
    why. This one does not, which is exactly why it is said here.

    A warning and not a refusal: the run is legitimate, it simply will not bank
    anything, and blocking it would refuse a launch the model accepts.
    """
    if not enabled(config):
        return ""
    thermal = _section(config, "thermal")
    if _truthy(thermal.get("run_thermal_mesh"), True):
        return ""
    return (
        "This run does not solve the thermal model, so nothing will be banked "
        "and the model will not mention it. Turn the thermal solution on, or "
        "turn the banking off, so the setting and the run agree.")


# ---------------------------------------------------------------------------
# 5. Size
# ---------------------------------------------------------------------------

def bytes_per_face(n_steps, n_layers=0, n_depths=0, with_column=False,
                   with_depths=False):
    """Bytes banked per facet -- the model's own ``bytesPerFace``.

    The surface temperature at every step is always banked. The final column is
    banked only when the run is saving a finished state file, and the
    subsurface accumulators only when subsurface maps are switched on; both are
    passed in explicitly rather than inferred from a zero, so "not banked" and
    "banked, and the count is not known here" stay distinguishable.
    """
    total = max(0, int(n_steps)) * SURFACE_BYTES_PER_STEP
    if with_column:
        total += max(0, int(n_layers)) * COLUMN_BYTES_PER_NODE
    if with_depths:
        total += max(0, int(n_depths)) * DEPTH_BYTES_PER_DEPTH + DEPTH_BYTES_FIXED
    return total


@dataclass
class SizeEstimate:
    """What the banked set will occupy, as far as it is knowable here.

    ``bytes_at_least`` counts the steps that are known from the configuration.
    It is a LOWER BOUND and ``exact`` is False whenever a term could not be
    counted -- which is the normal case, because the spin-up steps are settled
    by the solver.
    """
    bytes_at_least: int = 0
    exact: bool = False
    n_faces: int = 0
    n_steps_known: int = 0
    ceiling_bytes: int = 0
    unknown_terms: list = field(default_factory=list)

    @property
    def over_ceiling(self):
        return bool(self.ceiling_bytes) and self.bytes_at_least > self.ceiling_bytes


def size_estimate(config, n_faces, n_output_steps):
    """A lower bound on the banked set, and the ceiling it will be judged by.

    The count of steps that is knowable here is the reported window's. The
    spin-up is banked too and is NOT counted, because how long it runs is
    settled by the solver from the equilibration settings; that omission is
    named in ``unknown_terms`` rather than folded in as a guess, and it is why
    ``exact`` is False.
    """
    faces = max(0, int(n_faces or 0))
    steps = max(0, int(n_output_steps or 0))
    unknown = []

    if not faces:
        unknown.append(
            "how many facets the mesh will have, which is settled when the "
            "mesh is built")
    unknown.append(
        "the spin-up steps, which are banked as well and are settled by the "
        "solver from the equilibration settings")

    per_face = bytes_per_face(steps)

    thermal = _section(config, "thermal")
    if str(thermal.get("final_state_file") or "").strip():
        unknown.append(
            "four bytes per facet for every depth the model solves on, "
            "because this run also saves a finished state file")

    depths = (_section(config, "output") or {}).get("map_depths_m")
    if isinstance(depths, (list, tuple)) and depths:
        per_face += bytes_per_face(0, n_depths=len(depths), with_depths=True)

    gib = ceiling_gib(config)
    return SizeEstimate(
        bytes_at_least=per_face * faces,
        exact=False,
        n_faces=faces,
        n_steps_known=steps,
        ceiling_bytes=int(gib * GIB) if gib > 0 else 0,
        unknown_terms=unknown,
    )


def size_lines(config, n_faces, n_output_steps):
    """One or two sentences describing the size, for a read-out or a log."""
    if not enabled(config):
        return []
    est = size_estimate(config, n_faces, n_output_steps)
    if not est.n_faces or not est.n_steps_known:
        return [
            "Part-finished work is banked as the run goes. Its size is set by "
            "the number of facets and the number of steps, and neither is "
            "known yet; the model prints the exact figure when the run starts."]

    lines = ["Part-finished work: at least %s for %s facets over %s reported "
             "steps, and more once the spin-up steps are counted. The model "
             "prints the exact figure when the run starts."
             % (fmt_bytes(est.bytes_at_least), _commas(est.n_faces),
                _commas(est.n_steps_known))]
    if est.ceiling_bytes:
        if est.over_ceiling:
            lines.append(
                "That is already over the %s limit set below, so the run will "
                "stop before it starts. Raise the limit deliberately, or leave "
                "the banking off and split the run into shorter runs instead."
                % fmt_bytes(est.ceiling_bytes))
        else:
            lines.append(
                "The limit set below is %s, and the run stops rather than "
                "exceed it. The same figure is the extra memory a resumed run "
                "holds while it reads the banked work back."
                % fmt_bytes(est.ceiling_bytes))
    lines.append(
        "An interruption inside one group of %d facets loses that group's "
        "whole integration. To bound it more tightly than that, split the run "
        "into shorter runs." % BATCH_FACES)
    return lines


# ---------------------------------------------------------------------------
# 6. What is already banked
# ---------------------------------------------------------------------------

@dataclass
class BankedWork:
    """What is on disk in a partial directory.

    ``files_on_disk`` is counted from the directory. Everything named
    ``index_*`` is read from ``manifest.json``, which the model treats as an
    index and re-derives on every resume; the two are reported side by side so
    a disagreement is visible rather than resolved here.
    """
    directory: str = ""
    exists: bool = False
    files_on_disk: int = 0
    manifest_present: bool = False
    manifest_readable: bool = False
    index_files: int = 0
    index_batches: int = 0
    index_faces: int = 0
    index_faces_total: int = 0
    index_complete: bool = False
    #: Which parts of a facet group were banked, as the bit field the model
    #: writes. Bit 0 the surface series, bit 1 the finished column, bit 2 the
    #: subsurface accumulators.
    index_sections: int = 0
    index_depths: int = 0

    @property
    def any_work(self):
        return self.files_on_disk > 0

    @property
    def index_behind(self):
        """The normal shape of an interruption: a banked file landed and the
        index had not caught up when the run stopped."""
        return self.manifest_readable and self.index_files < self.files_on_disk


def scan_partials(directory):
    """Read a partial directory without judging it. Never raises."""
    out = BankedWork(directory=str(directory or ""))
    if not out.directory:
        return out
    try:
        names = os.listdir(out.directory)
    except OSError:
        return out
    out.exists = True
    out.files_on_disk = sum(1 for n in names if n.endswith(PARTIAL_SUFFIX))

    manifest = os.path.join(out.directory, MANIFEST_NAME)
    out.manifest_present = os.path.exists(manifest)
    if not out.manifest_present:
        return out
    try:
        with open(manifest, "r") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        # An index that cannot be read is not an error here. The model
        # rebuilds it from the banked files themselves, and says so.
        return out
    if not isinstance(data, dict):
        return out
    out.manifest_readable = True
    files = data.get("files")
    out.index_files = len(files) if isinstance(files, list) else 0
    out.index_batches = int(_number(data.get("batches_banked"), 0))
    out.index_faces = int(_number(data.get("faces_banked"), 0))
    out.index_faces_total = int(_number(data.get("n_faces_total"), 0))
    out.index_complete = _truthy(data.get("complete"), False)
    out.index_sections = int(_number(data.get("sections"), 0))
    out.index_depths = int(_number(data.get("n_depths"), 0))
    return out


#: The bits of the ``sections`` field, from the model.
SECTION_SURFACE_SERIES = 1 << 0
SECTION_FINAL_COLUMN = 1 << 1
SECTION_DEPTH_ACCUM = 1 << 2


def resume_conflicts(banked, config, n_faces=None):
    """Ways this configuration already disagrees with the banked work.

    Each of these is a field the model compares before it reads a banked file
    back, and a disagreement makes it refuse the whole run rather than mix two
    computations. Saying so before the launch turns a run that stops in its
    first seconds into a message beside the setting that caused it.

    Reported as warnings, never as refusals, and each says it is reading the
    INDEX. The model does not trust the index either -- it re-reads every
    banked file and compares that -- so a check written here against the index
    is a useful signal and not a verdict, and it must not block a launch on the
    strength of a file the model itself treats as advisory.
    """
    problems = []
    if banked is None or not banked.any_work or not banked.manifest_readable:
        return problems

    faces = int(n_faces or 0)
    if faces and banked.index_faces_total and banked.index_faces_total != faces:
        problems.append(
            "The index in %s records work banked for a mesh of %s facets, and "
            "this run expects %s. The model compares that and refuses to mix "
            "two meshes, so this run would stop. Point the banking at a "
            "different directory, or clear that one."
            % (banked.directory, _commas(banked.index_faces_total),
               _commas(faces)))

    wants_column = bool(str(
        _section(config, "thermal").get("final_state_file") or "").strip())
    had_column = bool(banked.index_sections & SECTION_FINAL_COLUMN)
    if banked.index_sections and wants_column != had_column:
        problems.append(
            "The index in %s records work banked %s a finished column per "
            "facet, and this run %s one. The model compares what each banked "
            "group contains and refuses when it differs, so this run would "
            "stop. That flag follows thermal.final_state_file."
            % (banked.directory,
               "with" if had_column else "without",
               "saves" if wants_column else "does not save"))

    depths = (_section(config, "output") or {}).get("map_depths_m")
    n_depths = len(depths) if isinstance(depths, (list, tuple)) else 0
    if banked.index_sections and n_depths != banked.index_depths:
        problems.append(
            "The index in %s records work banked for %d subsurface depth(s), "
            "and this run asks for %d. The model compares the depth product as "
            "well, and refuses when it differs, so this run would stop."
            % (banked.directory, banked.index_depths, n_depths))
    return problems


def resume_lines(banked):
    """What to tell someone about work already banked in a directory.

    Never an offer to resume, and never an offer to discard. The model reads
    banked work back whenever it finds it and verifies every file before it
    trusts one, so a control here would be a second, weaker opinion about a
    decision that has already been made strictly. What is useful is knowing
    that the work is there -- and, just as much, knowing that it is NOT, which
    is what a mistyped directory looks like.
    """
    if banked is None:
        return []
    if not banked.any_work:
        return [
            "No part-finished work was found in %s, so this run starts from "
            "the beginning. If you meant to carry on from an interrupted run, "
            "this is the wrong directory." % (banked.directory or "(unset)")]

    lines = ["%s holds %s part-finished file(s). The model checks each one "
             "against this run's mesh, depths, epoch, window and forcing, "
             "reads back the facet groups that match, and integrates the rest. "
             "It refuses the whole run if any of them disagrees."
             % (banked.directory, _commas(banked.files_on_disk))]
    if banked.manifest_readable and banked.index_faces_total:
        lines.append(
            "Its index reports %s of %s facets already finished across %s "
            "group(s)."
            % (_commas(banked.index_faces), _commas(banked.index_faces_total),
               _commas(banked.index_batches)))
    if banked.index_complete:
        # Measured: a run that finishes normally LEAVES its banked work in
        # place. Running the same configuration again then reads all of it back
        # and integrates nothing -- which is correct, and is not what someone
        # who has just changed something and pressed start is expecting to see.
        lines.append(
            "Its index reports the work as already complete. A run that "
            "finishes leaves its banked work behind, so starting this again "
            "would read all of it back and integrate nothing -- unless "
            "something the model checks has changed, in which case it stops. "
            "Clear the directory to integrate from the beginning.")
    if banked.index_behind:
        lines.append(
            "The index lists %s file(s) beside %s on disk. That is the usual "
            "shape of an interruption -- a file landed and the index had not "
            "caught up -- and the files are what count."
            % (_commas(banked.index_files), _commas(banked.files_on_disk)))
    elif banked.manifest_present and not banked.manifest_readable:
        lines.append(
            "Its index could not be read. The model rebuilds it from the "
            "files themselves, which are what count.")
    return lines


# ---------------------------------------------------------------------------
# 7. The launch check
# ---------------------------------------------------------------------------

@dataclass
class Findings:
    """What the launch check should report about this feature."""
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)
    line: str = ""

    @property
    def ok(self):
        return not self.errors


def preflight_findings(config, n_faces=None, n_frames=None,
                       output_directory=None):
    """Everything the launch check has to say about banking finished groups.

    Nothing at all when the feature is off, so a configuration that does not
    use it is not asked to read about it.
    """
    findings = Findings()
    if not enabled(config):
        return findings

    for blocker in engine_support(config).blockers:
        findings.errors.append("part-finished work: " + blocker)

    conflict = final_state_conflict(config, output_directory)
    if conflict:
        findings.errors.append("part-finished work: " + conflict)

    inert = inert_warning(config)
    if inert:
        findings.warnings.append("part-finished work: " + inert)

    directory = partial_dir(config, output_directory)
    banked = scan_partials(directory)
    for line in resume_lines(banked):
        findings.info.append("part-finished work: " + line)
    for problem in resume_conflicts(banked, config, n_faces):
        findings.warnings.append("part-finished work: " + problem)

    lines = size_lines(config, n_faces, n_frames)
    if lines:
        findings.line = " ".join(lines)
        est = size_estimate(config, n_faces, n_frames)
        if est.over_ceiling:
            # ⛔ An ERROR, not a note. The model stops on this, and a message
            # that reaches only a summary line while the launch proceeds is
            # the failure this project has already paid for once.
            findings.errors.append(
                "part-finished work: the banked set is at least %s, over the "
                "%s limit in thermal.batch_checkpoint_max_gib, and the model "
                "stops rather than exceed it."
                % (fmt_bytes(est.bytes_at_least),
                   fmt_bytes(est.ceiling_bytes)))
    return findings
