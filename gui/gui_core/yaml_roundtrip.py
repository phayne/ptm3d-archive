"""Comment-preserving YAML round trip for the GUI's save path (DP-1(b)).

The GUI's historical writer was ``yaml.dump`` of a merged dictionary. That is a
lossy operation twice over:

* it erases every comment (measured: 10 and 19 comments to 0 on two shipped
  examples, 185 to 0 on the 2026-08-14 tutorial config), and
* it cannot express **unset**. Writing every key it knows about turns
  ``gpu.min_vram_margin_gib`` -- the one key in the 184-key engine surface whose
  semantics depend on whether it was *written at all*
  (``config_parser.h``: presence makes the VRAM floor enforcing on every
  platform, not only macOS+AMD) -- into a universal guardrail for every config
  the GUI touches, on every machine.

This module is the other half of the fix. It keeps the loaded file's own text
and applies only the keys that actually changed, so:

* comments, ordering, blank lines and quoting survive;
* a key the file never mentioned and the user never edited is never written;
* the byte difference between the file before and after a save is confined to
  the lines whose values the user changed.

``ruamel.yaml`` is a HARD requirement of ``gui/requirements.txt``. When it is
missing the GUI does not silently fall back to the flattening writer without
saying so: ``UNAVAILABLE_MESSAGE`` is what the caller must show, and
``ConfigManager`` routes it to the console, the status bar and a one-time
dialog before the first lossy save.
"""

import io
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from ruamel.yaml import YAML
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    from ruamel.yaml.scalarstring import SingleQuotedScalarString
    RUAMEL_AVAILABLE = True
    RUAMEL_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - exercised by the absence test
    YAML = None
    CommentedMap = dict
    CommentedSeq = list
    SingleQuotedScalarString = str
    RUAMEL_AVAILABLE = False
    RUAMEL_IMPORT_ERROR = str(exc)


def quote_if_ambiguous(value):
    """Quote a string that a YAML 1.1 reader would resolve as something else.

    ruamel's round-trip emitter is YAML **1.2**, where `off` is a string.
    PyYAML -- which `load_from_yaml`, `test_config_parity` and every other
    tooling reader in this repo use -- is YAML **1.1**, where `off` is the
    boolean False. So writing the engine's own documented value
    `thermal.eq_indirect_source: off` bare meant the setting could not survive
    one save-and-reopen: it came back as False, `validate()` refused a config
    the engine accepts, and the run YAML carried `false`, which the engine
    FATALs on. The same split turned `bvh_cache_file: off` -- the DISABLE
    sentinel -- into a cache filename.

    PyYAML's own dumper already quotes these; this makes the round-trip writer
    agree with it. The test is general, not a list of words: anything whose
    text form does not read back as the same string gets quoted.
    """
    if not isinstance(value, str) or not RUAMEL_AVAILABLE:
        return value
    import yaml as _pyyaml
    try:
        if _pyyaml.safe_load(value) == value and isinstance(
                _pyyaml.safe_load(value), str):
            return value
    except Exception:                                  # noqa: BLE001
        pass
    return SingleQuotedScalarString(value)


def match_sequence_style(previous, value):
    """Keep a rewritten list in the layout the file already used for it.

    ruamel emits a plain Python list as a BLOCK sequence, so overwriting a
    value the file wrote inline --

        map_depths_m: [0.0, 0.05]

    -- turned one line into four. Those four lines DO carry the key the user
    edited, but `layout_only_changes` recognizes an edited line by matching a
    YAML key at the start of it and a sequence item has none, so the save also
    announced them as "layout the writer could not reproduce exactly": a
    correct edit, reported to the user as an unexplained reflow. Measured on a
    depth list and on `gpu.gpu_device_ids`, both of which shipped files spell
    inline.

    Only the FLOW case is handled, because block is already ruamel's own
    default: a key the file never carried has no style to copy and keeps
    whatever the emitter would have produced before this function existed.
    """
    if not RUAMEL_AVAILABLE or not isinstance(value, (list, tuple)):
        return value
    fa = getattr(previous, "fa", None)
    if fa is None or not fa.flow_style():
        return value
    seq = CommentedSeq(value)
    seq.fa.set_flow_style()
    return seq


UNAVAILABLE_MESSAGE = (
    "ruamel.yaml is not installed, so saving cannot preserve the comments and "
    "layout of the file it opened. Save will fall back to writing a flat, "
    "fully-expanded config: every comment in the file is lost, and every key "
    "the GUI knows about is written whether the file mentioned it or not "
    "(including gpu.min_vram_margin_gib, whose mere presence makes the VRAM "
    "floor enforcing on every platform). Install it with:  "
    "pip install -r gui/requirements.txt"
)


def available():
    """True when the comment-preserving save path can be used."""
    return RUAMEL_AVAILABLE


def detect_sequence_indent(text):
    """(sequence, offset) matching how `text` already indents block sequences.

    ruamel re-emits block sequences at ITS default (flush-left dashes under the
    parent key) unless told otherwise, so a file written as

        named_sites:
          - name: "Wilcoski"

    came back with the dashes at column 0 -- a byte difference on lines nobody
    edited. Measured on configs/survey_template.yaml: 6 rewritten lines on a
    save with an EMPTY edit plan.

    Returns ruamel's (sequence, offset) pair: `offset` is where the `-` sits,
    `sequence` is where the item's content sits. Defaults to ruamel's own
    (4, 2) shape only when the file shows no block sequence to copy.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue
        dash = len(line) - len(stripped)
        # The parent key's indentation: the nearest preceding non-blank,
        # non-comment line that is not itself a sequence item.
        for previous in reversed(lines[:i]):
            head = previous.strip()
            if not head or head.startswith("#") or head.startswith("- "):
                continue
            parent = len(previous) - len(previous.lstrip())
            offset = dash - parent
            if offset < 0:
                break
            # ruamel wants `sequence` >= `offset` + 2 ("- " is two columns).
            return (offset + 2, offset)
        break
    return (4, 2)


def _yaml(sequence=4, offset=2):
    y = YAML()                    # round-trip mode
    y.preserve_quotes = True
    # Do not re-wrap long scalars: a wrapped line is a byte difference on a key
    # nobody edited, which is exactly what this module exists to avoid.
    y.width = 4096
    y.indent(mapping=2, sequence=sequence, offset=offset)
    # PyYAML's safe_load takes the LAST of a duplicated key without complaint,
    # and the GUI has always loaded such files. ruamel raises by default, which
    # would turn a loadable config into "Failed to load project" -- a
    # regression introduced by a writer change. Match the reader.
    y.allow_duplicate_keys = True
    return y


def flatten(node, prefix=""):
    """Nested mapping -> {dotted key: leaf value}.

    A list is a LEAF (``gpu_device_ids: [0, 1]`` is one value, not two), which
    is also how the engine reads it.
    """
    out = {}
    if not isinstance(node, dict):
        return out
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, dotted + "."))
        else:
            out[dotted] = value
    return out


class RoundTripDoc:
    """A loaded YAML file, its original text, and its round-trip document."""

    def __init__(self, path, text, data):
        self.path = str(path) if path is not None else ""
        self.text = text
        self.data = data
        #: Copy the file's own block-sequence shape, so an edit to one scalar
        #: does not reflow a list somewhere else in the document.
        self.sequence, self.offset = detect_sequence_indent(text)

    @classmethod
    def load(cls, path):
        """Load `path` for round-trip editing, or return None.

        None means "the comment-preserving path is not available for this
        file" -- ruamel missing, or a document this parser cannot round-trip.
        The caller must fall back AND say that it did; it must not treat None
        as success.
        """
        if not RUAMEL_AVAILABLE:
            return None
        try:
            text = Path(path).read_text()
            data = _yaml().load(text)
        except Exception as exc:
            logger.warning(
                "[config] %s cannot be round-tripped (%s); saving it would "
                "fall back to the flattening writer and lose its comments.",
                path, exc)
            return None
        if not isinstance(data, dict):
            # A non-mapping document. load_from_yaml refuses these anyway; a
            # None here keeps the two paths agreeing about what the file is.
            return None
        return cls(path, text, data)

    def file_keys(self):
        """Dotted keys the FILE itself carries (not the merged defaults)."""
        return flatten(self.data)

    def apply(self, updates, removals=()):
        """Set `updates` (dotted key -> value) and delete `removals`.

        Intermediate mappings are created as needed, so a brand-new section
        (``gpu:``) can be added to a file that never had one. Removing a key
        does NOT prune the section that held it: the section header may carry
        the user's own comment, and deleting a comment nobody asked to delete
        is the behavior this module exists to stop.
        """
        for dotted, value in updates.items():
            parts = dotted.split(".")
            node = self.data
            for part in parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = CommentedMap()
                    node[part] = child
                node = child
            leaf = parts[-1]
            node[leaf] = match_sequence_style(
                node.get(leaf) if isinstance(node, dict) else None,
                quote_if_ambiguous(value))

        for dotted in removals:
            parts = dotted.split(".")
            node = self.data
            for part in parts[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
                if not isinstance(node, dict):
                    node = None
                    break
            if isinstance(node, dict) and parts[-1] in node:
                del node[parts[-1]]

    def dump(self):
        """Render the document back to text, in the file's own layout."""
        buf = io.StringIO()
        _yaml(self.sequence, self.offset).dump(self.data, buf)
        return buf.getvalue()


def values_equal(a, b):
    """Is `b` the SAME setting as `a`, for the purpose of "did this change"?

    Plain `==` is not enough and plain `type(a) is type(b)` is too much.

    * ``1`` and ``1.0`` are the same setting. A file that spells a
      double-valued key as an integer -- ``scale_factor: 1``,
      ``geothermal_heat_flow: 0`` -- gets that value coerced to float by
      ``set_numeric`` when the panel loads it, and a strict type test then
      rewrote both lines on a save with NO EDITS AT ALL. Reproduced.
    * ``0`` and ``False`` are NOT the same setting, even though Python says
      ``0 == False``. A config spelling a switch as ``0`` that the user then
      ticks must be written, and a value test alone would miss it.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


def layout_only_changes(before, after, edited_keys):
    """Changed lines that mention none of `edited_keys` -- i.e. pure reflow.

    `detect_sequence_indent` copies the FIRST block sequence's shape, which is
    all ruamel offers: there is no per-node indentation. A document that mixes
    shapes will have the non-matching ones reflowed. That is a real limitation,
    so it is measured and reported instead of assumed away.
    """
    import difflib
    import re
    # Every segment, not just the leaf: writing `simulation.timesteps` into a
    # file that had no `simulation:` legitimately adds that header, and calling
    # the header a reflow would cry wolf on a line the edit really did create.
    #
    # Matched as a YAML KEY, anchored, not as a bare substring. `leaf in line`
    # let an edit to output.directory absorb a reflow of `topo_directory:`,
    # and segments like `thermal`, `mesh` and `body` made that common -- so the
    # guard rail went quiet exactly where it was needed.
    segments = set()
    for key in edited_keys:
        segments.update(key.split("."))
    if not segments:
        pattern = None
    else:
        pattern = re.compile(
            r"^[+-]\s*(?:- )?(" + "|".join(re.escape(x) for x in sorted(segments))
            + r")\s*:")
    changed = [
        line for line in difflib.unified_diff(
            before.splitlines(), after.splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))]
    if pattern is None:
        changed = list(changed)
    else:
        changed = [line for line in changed if not pattern.match(line)]
    # A REMOVED comment or blank line is a consequence of removing the key it
    # annotated, not a reflow. Clearing a custom body deletes
    # generic_body.orbit.* and the comments above them, and reporting those as
    # "reformatted, carrying no key you edited" is the guard rail crying wolf
    # on a correct deletion. An ADDED comment would be a real surprise and is
    # still reported.
    def _is_removed_annotation(line):
        if not line.startswith("-"):
            return False
        body = line[1:].strip()
        return body == "" or body.startswith("#")

    return [line for line in changed if not _is_removed_annotation(line)]


def plan_edits(loaded_view, current_view, file_keys):
    """Decide what a save has to change, and nothing more.

    ``loaded_view``  the engine-facing view of the config AS LOADED
    ``current_view`` the engine-facing view of the config NOW
    ``file_keys``    the dotted keys the file itself carries

    Returns ``(updates, removals)``.

    The rule that makes "unset" representable: a key whose current value equals
    what it had at load time is not written, even when the GUI has a default
    for it and the file does not. So a file that never mentioned ``gpu:`` still
    does not mention it unless the user changed something under it.
    """
    was = flatten(loaded_view)
    now = flatten(current_view)

    updates = {}
    for key, value in now.items():
        if key not in was or not values_equal(was[key], value):
            updates[key] = value

    # A key that left the engine-facing view (an override the user unticked)
    # is removed only if the file actually carries it.
    removals = [key for key in was if key not in now and key in file_keys]
    return updates, removals
