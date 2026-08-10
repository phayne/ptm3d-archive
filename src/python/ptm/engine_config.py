"""What the engine actually reads from a YAML config, and with what default.

``ConfigParser::loadFromYAML`` (``src/cpp/tools/config_parser.cpp``) is the only
authority on which keys exist. Every key goes through ``getWithDefault``, there
is no required key, and an empty YAML file is a valid config -- so a
misspelled key is not an error, it is silence. Anything on the Python side that
wants to know "is this a real key?" or "what happens if I leave it out?" has,
until now, had to answer from documentation that covers 87 of the keys.

This module answers from the source:

* :func:`parse_engine_config` reads ``config_parser.h`` and
  ``config_parser.cpp`` and returns ``{dotted key: default}`` for every key the
  parser consumes.
* :data:`ENGINE_DEFAULTS` in ``ptm/engine_defaults.py`` is that result,
  committed, so the table is available without the C++ sources at hand.
* ``src/python/tests/test_config_parity.py`` re-derives the table from the
  sources and fails if it no longer matches the committed one. A default that
  drifts in C++ therefore breaks a test instead of quietly changing what the
  Python layer believes.

Regenerate after an intentional engine change::

    python3 -m ptm.engine_config --regenerate

Limits worth stating: the parse is textual, so it tracks the shape of
``loadFromYAML`` as written today (``getWithDefault(node, "key", ...)`` inside
``if (root["section"])`` blocks). If that shape changes, the parse loses keys
-- which is why the test asserts a floor on the key count and on the set of
sections, instead of trusting whatever it happens to find.
"""

from __future__ import annotations

import argparse

import re
import sys
from pathlib import Path

from .engine_defaults import ENGINE_DEFAULTS, ENGINE_SECTIONS, KEYS_WITHOUT_DEFAULT

__all__ = [
    "ENGINE_DEFAULTS",
    "ENGINE_SECTIONS",
    "KEYS_WITHOUT_DEFAULT",
    "default_for",
    "flatten_config",
    "get_dotted",
    "known_keys",
    "parse_engine_config",
    "set_dotted",
    "unknown_keys",
]

_HEADER = "src/cpp/tools/config_parser.h"
_SOURCE = "src/cpp/tools/config_parser.cpp"

# C++ spellings of "empty string" / "empty container" used as defaults.
_EMPTY_LITERALS = {"std::string{}", "std::string()", 'std::string("")', '""', "{}"}


# --------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------

def known_keys():
    """Every dotted key the engine's YAML parser consumes."""
    return set(ENGINE_DEFAULTS)


def default_for(key):
    """The engine's default for ``key``. Raises ``KeyError`` if unknown.

    ``None`` means the key exists but has no literal default in the parser
    (a sequence, or a value read outside ``getWithDefault``); those keys are
    listed in :data:`KEYS_WITHOUT_DEFAULT`.
    """
    return ENGINE_DEFAULTS[key]


def get_dotted(config, dotted):
    """``get_dotted(cfg, 'thermal.use_gpu')``, or ``None`` if absent.

    Here rather than in the caller because the friction audit counted **seven
    dialects** of "set a nested config key by path" across nineteen sweep
    scripts, and this is the config module.
    """
    node = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_dotted(config, dotted, value):
    """Set ``cfg['a']['b'] = value`` from ``'a.b'``, creating dicts as needed."""
    parts = dotted.split(".")
    node = config
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def flatten_config(config, prefix=""):
    """Flatten a loaded YAML config to ``{dotted key: value}``.

    Mapping values are descended into; everything else is a leaf. ``engine_env``
    is returned whole, because it is a free-form table of environment variables
    rather than a set of known keys.
    """
    out = {}
    for key, value in (config or {}).items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if dotted == "engine_env":
            out[dotted] = value
        elif isinstance(value, dict):
            out.update(flatten_config(value, dotted))
        else:
            out[dotted] = value
    return out


def unknown_keys(config):
    """Dotted keys in ``config`` that the engine's parser never reads.

    The engine does not reject these; it does not see them at all. Sorted.
    """
    known = known_keys()
    unknown = []
    for dotted in flatten_config(config):
        if dotted == "engine_env":
            continue
        if dotted in known:
            continue
        # A parent that is itself a known key's prefix is a section, not a leaf.
        if any(k.startswith(dotted + ".") for k in known):
            continue
        unknown.append(dotted)
    return sorted(unknown)


# --------------------------------------------------------------------------
# Deriving the table from the C++ sources
# --------------------------------------------------------------------------

def _strip_comments(text):
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"/\*.*?\*/", "", text, flags=re.S)


def _cxx_literal(token):
    """Convert a C++ default-initializer token to a Python value.

    Returns the pair ``(ok, value)``; ``ok`` is False for anything that is not
    a plain literal, so the caller can report it rather than guess.
    """
    if token is None:
        return True, None
    tok = token.strip()
    if tok in _EMPTY_LITERALS:
        return True, ""
    if tok in ("true", "false"):
        return True, tok == "true"
    match = re.fullmatch(r'"((?:[^"\\]|\\.)*)"', tok)
    if match:
        return True, match.group(1).encode().decode("unicode_escape")
    if re.fullmatch(r"[+-]?\d+", tok):
        return True, int(tok)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?f?", tok):
        return True, float(tok.rstrip("f"))
    return False, tok


def _parse_struct_defaults(header_text):
    """``{'time.body': ('std::string', '"Europa"'), ...}`` from the header."""
    start = header_text.index("struct SimulationConfig")
    brace = header_text.index("{", start)
    depth = 0
    body = None
    for i in range(brace, len(header_text)):
        if header_text[i] == "{":
            depth += 1
        elif header_text[i] == "}":
            depth -= 1
            if depth == 0:
                body = header_text[brace + 1:i]
                break
    if body is None:
        raise ValueError(f"{_HEADER}: struct SimulationConfig is not brace-balanced")

    frames = [{}]
    for raw in _strip_comments(body).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("struct ") and line.endswith("{"):
            frames.append({})
            continue
        if line.startswith("}") and line.endswith(";") and len(frames) > 1:
            instance = line[1:-1].strip()
            closed = frames.pop()
            for name, value in closed.items():
                frames[-1][f"{instance}.{name}" if instance else name] = value
            continue
        if not line.endswith(";"):
            continue
        statement = line[:-1]
        if "=" in statement:
            decl, default = statement.split("=", 1)
            default = default.strip()
        else:
            decl, default = statement, None
        tokens = decl.replace("*", " ").split()
        if len(tokens) < 2:
            continue
        name = tokens[-1]
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        frames[-1][name] = (" ".join(tokens[:-1]), default)
    return frames[0]


_TOKEN_RE = re.compile(
    # const auto& alias = <node>["section"];
    r'const\s+auto&\s+(\w+)\s*=\s*(\w+)\["([^"]+)"\]'
    # getWithDefault(node, "key", config.<struct path>)
    r'|getWithDefault(?:<[^>]*>)?\(\s*(\w+)\s*,\s*"([^"]+)"\s*,\s*config\.([\w.]+)\s*\)'
    # getWithDefault(node, "key", <literal>)
    r'|getWithDefault(?:<[^>]*>)?\(\s*(\w+)\s*,\s*"([^"]+)"\s*,'
    r'\s*([^,()]*?(?:\([^()]*\)|\{[^{}]*\})?[^,()]*?)\s*\)'
    # node["key"] read directly (sequences, size_t, sub-nodes)
    r'|(\w+)\["([^"]+)"\]'
)


def parse_engine_config(repo_root):
    """Derive the engine's config surface from the C++ parser sources.

    Returns ``(defaults, no_default, unparsed)``:

    * ``defaults``   ``{dotted key: python value}`` for every leaf key
    * ``no_default`` sorted keys read without a literal default
    * ``unparsed``   ``[(key, c++ token)]`` defaults that are not plain
      literals -- empty on a healthy parse, and something the caller should
      surface rather than swallow
    """
    root = Path(repo_root)
    header = (root / _HEADER).read_text()
    source = (root / _SOURCE).read_text()

    struct_defaults = _parse_struct_defaults(header)

    start = source.index("SimulationConfig ConfigParser::loadFromYAML")
    body = _strip_comments(source[start:])

    alias = {}
    sections = set()
    found = {}
    for match in _TOKEN_RE.finditer(body):
        if match.group(1):
            base = match.group(2)
            prefix = "" if base == "root" else alias.get(base, "?" + base)
            full = (prefix + "." if prefix else "") + match.group(3)
            alias[match.group(1)] = full
            sections.add(full)
        elif match.group(4):
            node = alias.get(match.group(4), "?" + match.group(4))
            found.setdefault(f"{node}.{match.group(5)}", ("struct", match.group(6)))
        elif match.group(7):
            node = alias.get(match.group(7), "?" + match.group(7))
            found.setdefault(f"{node}.{match.group(8)}", ("literal", match.group(9).strip()))
        elif match.group(10) and match.group(10) in alias:
            found.setdefault(f"{alias[match.group(10)]}.{match.group(11)}", ("none", None))

    unresolved = sorted({k.split(".")[0] for k in found if k.startswith("?")})
    if unresolved:
        raise ValueError(
            f"{_SOURCE}: could not resolve the YAML section for node alias(es) "
            f"{unresolved}; the parse would silently drop their keys"
        )

    defaults = {}
    no_default = []
    unparsed = []
    for key in sorted(k for k in found if k not in sections):
        kind, ref = found[key]
        if kind == "struct":
            if ref not in struct_defaults:
                raise ValueError(
                    f"{_SOURCE}: '{key}' defaults to config.{ref}, which is not a "
                    f"field of SimulationConfig in {_HEADER}"
                )
            ok, value = _cxx_literal(struct_defaults[ref][1])
        elif kind == "literal":
            ok, value = _cxx_literal(ref)
        else:
            ok, value = True, None
            no_default.append(key)
        if not ok:
            unparsed.append((key, value))
            value = None
        defaults[key] = value

    return defaults, sorted(no_default), unparsed


# --------------------------------------------------------------------------
# Regeneration
# --------------------------------------------------------------------------

_GENERATED_HEADER = '''\
"""Engine config keys and defaults -- GENERATED, do not edit by hand.

Derived from src/cpp/tools/config_parser.{{h,cpp}} by

    python3 -m ptm.engine_config --regenerate

src/python/tests/test_config_parity.py re-derives this from the C++ sources on
every run and fails if the two disagree, so an engine-side change to a default
shows up as a test failure and a diff to this file rather than as a Python
layer that quietly believes the old value.

{count} keys across {nsections} sections.
"""

#: Sections the parser reads at the top level of the YAML.
ENGINE_SECTIONS = {sections}

#: Keys read without a literal default: sequences, a size_t read directly, and
#: one nested list. Their entry in ENGINE_DEFAULTS is None, which means "no
#: default in the parser", not "defaults to null".
KEYS_WITHOUT_DEFAULT = {nodefault}

#: {{dotted YAML key: engine default}}. Every key is optional -- the parser has
#: no required key and an empty config file is valid.
ENGINE_DEFAULTS = {defaults}
'''


def _render_tuple(items):
    if not items:
        return "()"
    return "(\n" + "".join(f"    {item!r},\n" for item in items) + ")"


def _render_dict(mapping):
    if not mapping:
        return "{}"
    return "{\n" + "".join(
        f"    {key!r}: {mapping[key]!r},\n" for key in sorted(mapping)
    ) + "}"


def _render_module(defaults, no_default):
    sections = sorted({k.split(".")[0] for k in defaults})
    return _GENERATED_HEADER.format(
        count=len(defaults),
        nsections=len(sections),
        sections=_render_tuple(sections),
        nodefault=_render_tuple(no_default),
        defaults=_render_dict(defaults),
    )


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate or check ptm/engine_defaults.py against the C++ parser.")
    parser.add_argument("--repo-root", default=None,
                        help="PTM3D checkout (default: found by walking up from this file)")
    parser.add_argument("--regenerate", action="store_true",
                        help="rewrite ptm/engine_defaults.py from the C++ sources")
    args = parser.parse_args(argv)

    from . import paths
    root = Path(args.repo_root) if args.repo_root else paths.find_repo_root(Path(__file__))
    if root is None:
        parser.error("no PTM3D checkout found; pass --repo-root")

    defaults, no_default, unparsed = parse_engine_config(root)
    for key, token in unparsed:
        sys.stderr.write(f"[ptm] {key}: default {token!r} is not a plain literal\n")

    target = Path(__file__).with_name("engine_defaults.py")
    rendered = _render_module(defaults, no_default)
    if args.regenerate:
        target.write_text(rendered)
        print(f"wrote {target} ({len(defaults)} keys)")
        return 0

    current = target.read_text() if target.exists() else ""
    if current == rendered:
        print(f"engine_defaults.py is current ({len(defaults)} keys)")
        return 0
    print("engine_defaults.py is STALE; rerun with --regenerate", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
