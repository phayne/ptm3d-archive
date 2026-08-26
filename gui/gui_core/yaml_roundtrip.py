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
    from ruamel.yaml.comments import CommentedMap
    RUAMEL_AVAILABLE = True
    RUAMEL_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - exercised by the absence test
    YAML = None
    CommentedMap = dict
    RUAMEL_AVAILABLE = False
    RUAMEL_IMPORT_ERROR = str(exc)


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


def _yaml():
    y = YAML()                    # round-trip mode
    y.preserve_quotes = True
    # Do not re-wrap long scalars: a wrapped line is a byte difference on a key
    # nobody edited, which is exactly what this module exists to avoid.
    y.width = 4096
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
            node[parts[-1]] = value

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
        """Render the document back to text."""
        buf = io.StringIO()
        _yaml().dump(self.data, buf)
        return buf.getvalue()


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
        if key not in was or was[key] != value or type(was[key]) is not type(value):
            updates[key] = value

    # A key that left the engine-facing view (an override the user unticked)
    # is removed only if the file actually carries it.
    removals = [key for key in was if key not in now and key in file_keys]
    return updates, removals
