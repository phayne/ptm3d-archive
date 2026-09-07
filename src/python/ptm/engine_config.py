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

One textual ambiguity is resolved rather than assumed away.
``const auto& X = Y["Z"];`` is written for both a section and a single key, and
reading every one of them as a section dropped the keys bound that way while
still reporting the table current (PTM-ENGINECONFIG-ALIAS-DROPS-KEY). See
:func:`_section_aliases`: an alias is a section only if the parser goes on to
read keys OUT of it.
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

#: Just the binding, for the pre-scan that decides what each one MEANS.
_BIND_RE = re.compile(r'const\s+auto&\s+(\w+)\s*=\s*(\w+)\["([^"]+)"\]')

#: "not in the middle of an identifier". `\b` is wrong here: it would still let
#: `d` match the tail of `rad`, because `d` is a word character on both sides.
_WORD_L = r'(?<![A-Za-z0-9_])'


def _balanced_body(text, start):
    """``text`` from ``start`` to the close of the first brace-balanced block.

    Falls back to the rest of the file if the braces do not balance, so a parse
    that cannot find the end still sees at least what it saw before -- the
    caller's other guards (the key-count floor, the section set) are what catch
    a genuinely broken parse, and losing everything here would just move the
    failure somewhere less legible.
    """
    open_at = text.find("{", start)
    if open_at < 0:
        return text[start:]
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _section_aliases(body):
    """Names bound by ``const auto& X = Y["Z"];`` that are really SECTIONS.

    ``const auto&`` says nothing about what was bound. The parser uses it for
    two different things:

        const auto& output = root["output"];        a SECTION -- Z is a section
                                                    name and its keys follow
        const auto& d = output["map_depths_m"];     a KEY -- Z is the key, and
                                                    the alias is the value

    Reading both as the first was PTM-ENGINECONFIG-ALIAS-DROPS-KEY: the second
    form put ``output.map_depths_m`` in the SECTIONS set, and sections are
    excluded from the defaults table, so the key was dropped -- while
    ``python3 -m ptm.engine_config`` still printed "engine_defaults.py is
    current". A table that silently loses a key while reporting itself current
    is worse than one that fails, because everything downstream (the GUI's
    key check, the dead-key audit) then agrees the key does not exist.

    The distinguishing fact is USE, not spelling: a section is a node you go on
    to read keys OUT of, either as ``getWithDefault(X, "k", ...)`` or as
    ``X["k"]``. A bound KEY is used as a value -- iterated, ``.as<T>()``,
    ``.IsSequence()`` -- and never subscripted with a key name.

    A name bound twice is a hard error rather than a guess: this returns one
    verdict per NAME, so two bindings with different roles cannot both be
    honored, and picking one silently is the class of failure this function
    exists to close.
    """
    bound = {}
    for match in _BIND_RE.finditer(body):
        name = match.group(1)
        if name in bound:
            raise ValueError(
                f"{_SOURCE}: the node alias {name!r} is bound more than once "
                f"inside loadFromYAML ({bound[name]!r} and "
                f"{match.group(3)!r}). This extractor classifies an alias by "
                f"name, so it cannot tell the two apart; rename one.")
        bound[name] = match.group(3)
    # ⛔ `_WORD_L` is not decoration. Without it, `d["` matches inside `rad["`,
    # and `config_parser.cpp` really does contain `rad["`, `gb["`, `spin["`,
    # `mesh["`, `thermal["`, `shape["` and `output["` -- so an alias named `d`,
    # `b`, `n`, `e`, `h` or `l` would be classified as a SECTION by a subscript
    # written somewhere else entirely, and its key would be dropped exactly as
    # before. Found in review, reproduced against the shipped source shape, and
    # pinned by the second half of
    # test_config_parity.test_an_aliased_key_is_not_read_as_a_section.
    used_as_section = set()
    for name in bound:
        escaped = _WORD_L + re.escape(name)
        if re.search(r'getWithDefault(?:<[^>]*>)?\(\s*' + escaped + r'\s*,', body) \
                or re.search(escaped + r'\["', body):
            used_as_section.add(name)
    return used_as_section


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

    # ⛔ The BRACE-BALANCED function, not "everything after it". The tail also
    # holds saveToYAML, validate, overrideFromCommandLine and the no-yaml
    # fallback; reading those was harmless while the extractor only collected
    # matches, but `_section_aliases` now REFUSES on a duplicate alias name, and
    # a future `const auto& thermal = ...` in saveToYAML would hard-fail the
    # whole parity suite over legal C++ in a function this tool does not model.
    # Measured: scoping changes no key (189 before and after) -- so this is a
    # correction of what the docstring already claimed, not a behavior change.
    start = source.index("SimulationConfig ConfigParser::loadFromYAML")
    body = _strip_comments(_balanced_body(source, start))

    alias = {}
    sections = set()
    found = {}
    section_aliases = _section_aliases(body)
    for match in _TOKEN_RE.finditer(body):
        if match.group(1):
            base = match.group(2)
            prefix = "" if base == "root" else alias.get(base, "?" + base)
            full = (prefix + "." if prefix else "") + match.group(3)
            if match.group(1) in section_aliases:
                alias[match.group(1)] = full
                sections.add(full)
            else:
                # A KEY bound to a name, not a section. Recorded as the direct
                # read it is; the alias is deliberately NOT registered, so a
                # later `d["something"]` on it cannot invent a nested key.
                found.setdefault(full, ("none", None))
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
