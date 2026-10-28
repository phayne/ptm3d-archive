"""Reachable configuration presets (item 2.3 / DP-5).

``ConfigManager.PRESETS`` used to hold five of these and nothing ever called
them -- ``grep -rn load_preset gui/`` returned the definition and nothing else
-- while ``gui/README.md`` advertised the feature. Stage 1 deleted the dead
table rather than pretend it worked (GUI-PRESET-1); this is the replacement,
built from §4.4 of ``the GUI update plan (docs/design/gui-update-plan.md in git history)``.

Two rules govern the whole module:

1. **A preset states what it set.** ``apply()`` returns the exact dotted keys
   and values it wrote, and the GUI shows them. A preset that quietly rewrote
   half a config would be the same defect as the old save path.

2. **A preset never sets ``radiation.beaming`` past the facet-scale rule.**
   The engine enforces only that ``beaming: true`` carries a positive slope and
   NOTHING about the scale of the mesh it is applied to, so the GUI is the only
   thing between a user and double-counted roughness
   (``docs/radiation-exchange.md`` §8.4). A preset that asks for beaming
   evaluates ``gui_core.facet_scale.beaming_policy`` on the config it just
   built, and REFUSES -- with the measured facet scale in the message -- rather
   than turning the switch on and hoping.
"""

from dataclasses import dataclass, field


#: The validated lunar polar RMS sub-facet slope (Jhoti et al. 2024, Table 1).
LUNAR_RMS_SLOPE_DEG = 32.5


#: What a BUILT-IN body's card clears: the whole `generic_body:` section.
#:
#: The engine reads `generic_body:` first and never consults `simulation.body`
#: when one is configured -- `isConfigured()` is `!name.empty() || ...`
#: (config_parser.h:717), tested at ptm_thermal_sim.cpp:14626. So a single
#: leftover name under `generic_body:` does not merely add a stray key; it
#: decides which body runs. Measured 2026-08-27: a name typed on the Asteroid
#: card, then the Europa Global card, then Run, produced a configuration
#: reading `simulation.body: Europa` whose run recorded `body=bennu`,
#: `generic_body=true` and a 99.99 km median radius.
#:
#: Clearing it also restores the `thermophysical:` cleanup, which
#: `_config_for_yaml` gates on the same "is a generic body active" question --
#: that is why the same run also emitted the dead 19-key thermophysical block
#: (PTM-67) instead of the `*_override` keys.
#:
#: `illumination.include_primary_reflection` is here for the mirror-image
#: reason. The generic-body cards set it FALSE, which is right for a
#: heliocentric asteroid and wrong for a moon: Europa's night side is lit by
#: Jupiter, and leaving the switch off carried that straight into the next
#: Europa run. It is a body-dependent default, so a body card must state it
#: rather than inherit it.
BUILTIN_BODY_CLEARS = (
    "generic_body",
    "illumination.include_primary_reflection",
)

#: What a GENERIC body's card clears: another body's mesh placement.
#:
#: The mirror of the above, and the same engine rule underneath -- the solver
#: takes `mesh.import_file` whenever it is non-empty and reaches the icosphere
#: branch only when it is empty (ptm_thermal_sim.cpp:15087). A lunar crater
#: core mesh left behind by the Moon card would therefore become the asteroid's
#: shape, at a latitude and longitude that mean nothing on it, against a
#: view-factor matrix built for a different mesh entirely.
#:
#: The `generic_body:` section is NOT cleared here: it is what these cards are
#: for. Nor is `mesh.resolution`, which the generic cards use for their own
#: procedural sphere.
#:
#: `mesh.import_file` and `mesh.scale_factor` ARE cleared, and then
#: :func:`mirror_shape_mesh` puts this body's OWN shape model back. Clearing
#: without that step would delete a `custom_mesh` asteroid's DAMIT model:
#: `mesh.import_file` is a MIRROR of `generic_body.shape.mesh_file` (the solver
#: reads the top-level path, the generic-body one is metadata), and the card's
#: `_propagate_shape_mesh` early-returns while a card is loading, so nothing
#: would re-establish it. The run would then use the procedural sphere from
#: `mesh.resolution` with the shape model unread on disk -- silently.
GENERIC_BODY_CLEARS = (
    "mesh.import_file",
    "mesh.scale_factor",
    "mesh.occluder_file",
    "mesh.latitude_override",
    "mesh.longitude_override",
    "radiation.view_factor_file",
)

#: The mesh keys that describe WHERE ON WHICH BODY a local run sits, cleared
#: when the user changes body.
#:
#: NOT part of any preset's `clears`. The global cards already reset these
#: through their `sets` (a global run has no imported mesh), and a local-mesh
#: card must NOT reset them just because it was opened: the lunar crater card
#: is about an imported mesh, and clearing it on every open would leave the
#: beaming facet-scale rule with nothing to measure on a mesh the user had
#: already chosen.
#:
#: The rule is about the BODY changing, so it is applied where the body change
#: is known -- `BodyFirstPanel._apply_route_defaults`. The lunar crater card
#: and the Europa DEM card both write `mesh.import_file`,
#: `mesh.latitude_override` and `radiation.view_factor_file`, and going from
#: one to the other used to carry the crater mesh, its lunar latitude and a
#: view-factor matrix built for a different mesh into a Europa run.
FOREIGN_MESH_CLEARS = (
    "mesh.import_file",
    "mesh.occluder_file",
    "mesh.latitude_override",
    "mesh.longitude_override",
    "radiation.view_factor_file",
)

#: The shape spelling the engine branches on for a mesh-shaped generic body
#: (`ptm_thermal_sim.cpp:14707`). Duplicated from `body_cards.GENERIC_MESH_SHAPE`
#: only in the sense that both are pinned to the C++ literal by
#: `test_gui_body_first.py`; imported rather than retyped below.
def mirror_shape_mesh(config_manager):
    """Point `mesh.import_file` at this generic body's own shape model.

    ONE rule, in one place. `GenericBodyCard._propagate_shape_mesh` applies the
    same rule when the user picks a shape file, and this applies it when a card
    has just cleared the mesh keys. Two copies of "which key the solver reads"
    is how the mirror came to be missing on one of the two paths.

    Returns ``{dotted key: value}`` for what it wrote, or ``{}``.
    """
    shape_type = str(config_manager.get_value(
        "generic_body.shape.type", "") or "").strip()
    from gui_core.body_cards import GENERIC_MESH_SHAPE
    if shape_type != GENERIC_MESH_SHAPE:
        return {}
    mesh_file = str(config_manager.get_value(
        "generic_body.shape.mesh_file", "") or "").strip()
    if not mesh_file:
        return {}
    scale = config_manager.get_value("generic_body.shape.mesh_scale", 1.0)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 1.0
    written = {"mesh.import_file": mesh_file, "mesh.scale_factor": scale}
    for key, value in written.items():
        config_manager.set_value(key, value)
    return written


@dataclass
class PresetResult:
    """What applying a preset actually did."""
    applied: dict = field(default_factory=dict)   # dotted key -> value written
    refused: list = field(default_factory=list)   # (dotted key, reason)
    notes: list = field(default_factory=list)     # things the user must supply
    #: dotted key -> the value that was there, for each key a `clears` section
    #: reset. Reported, because a section quietly reset is the same defect as a
    #: section quietly kept.
    cleared: dict = field(default_factory=dict)

    def summary_lines(self):
        out = [f"Set {len(self.applied)} key(s):"]
        for key in sorted(self.applied):
            out.append(f"    {key} = {self.applied[key]!r}")
        if self.cleared:
            out.append(f"Cleared {len(self.cleared)} key(s) left by another "
                       f"body's card:")
            for key in sorted(self.cleared):
                out.append(f"    {key} was {self.cleared[key]!r}")
        for key, reason in self.refused:
            out.append(f"REFUSED {key}: {reason}")
        for note in self.notes:
            out.append(f"NOTE: {note}")
        return out


@dataclass
class Preset:
    key: str
    title: str
    summary: str
    #: Dotted key -> value, applied in order.
    sets: dict = field(default_factory=dict)
    #: Whole top-level SECTIONS this preset resets to the engine default,
    #: because carrying them under this body would describe a different body.
    #:
    #: `sets` cannot do this job. It writes the keys a card shows, and the
    #: defect it left open is exactly the one measured on 2026-08-27: a name
    #: typed on the Asteroid card survived the trip back to the landing screen,
    #: the Europa card wrote `simulation.body: Europa` over the top of it, and
    #: the engine -- which reads `generic_body:` FIRST and never looks at
    #: `simulation.body` when one is present (`ptm_thermal_sim.cpp:14626`) --
    #: ran a 100 km sphere at 1 AU. Clearing by SECTION rather than by key list
    #: means a key added to `generic_body:` later is cleared too.
    clears: tuple = ()
    #: "on" (subject to the scale rule), "off", or "n/a" (radiation is off).
    beaming: str = "off"
    #: Fields the user still has to fill in for this to be a runnable config.
    needs: tuple = ()
    #: True when at least one of `needs` lives behind the Advanced disclosure.
    #: The card then turns Advanced on, because telling a Basic-tier user to
    #: fill in a field the Basic tier hides is the exact failure the tiering
    #: rework exists to remove -- and it was reproduced: with the toggle off,
    #: mesh.occluder_file, both placement overrides and
    #: radiation.view_factor_file are all hidden.
    needs_advanced: bool = False
    #: Why this preset does not offer beaming, when it does not.
    beaming_note: str = ""


PRESETS = [
    Preset(
        key="lunar_global_diurnal",
        title="Global lunar diurnal",
        summary=(
            "A whole-Moon icosphere over one synodic period, radiation "
            "exchange off. The standard first run, and the one every other "
            "preset is a departure from."),
        sets={
            "simulation.body": "Moon",
            "simulation.duration_days": 29.530588,
            "simulation.timesteps": 144,
            "mesh.resolution": 5,
            "mesh.import_file": "",
            "mesh.occluder_file": "",
            "mesh.latitude_override": -999.0,
            "mesh.longitude_override": -999.0,
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "builtin",
        },
        clears=BUILTIN_BODY_CLEARS,
        beaming="n/a",
        beaming_note=(
            "Radiation exchange is off, and the engine reads radiation.beaming "
            "only on the radiation path -- the switch would be inert."),
    ),
    Preset(
        key="lunar_polar_crater",
        title="Lunar polar crater with self-heating",
        summary=(
            "An imported high-resolution core mesh plus its occluder ring, "
            "radiation exchange on in bounded coupling mode, and Smith (1967) "
            "sub-facet roughness at the validated lunar polar slope -- IF the "
            "mesh is coarse enough to carry it."),
        sets={
            "simulation.body": "Moon",
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": True,
            "radiation.coupling_mode": "bounded",
            "radiation.nearfield_quadrature_order": 3,
            "illumination.albedo_source": "builtin",
        },
        # The mesh keys are NOT cleared here: this card is about an imported
        # mesh, and the beaming rule below has to be able to measure one the
        # user has already chosen. A DIFFERENT body's terrain is cleared by
        # FOREIGN_MESH_CLEARS, on the body change, where that is knowable.
        clears=BUILTIN_BODY_CLEARS,
        beaming="on",
        needs_advanced=True,
        needs=(
            "mesh.import_file -- the high-resolution core mesh",
            "mesh.occluder_file -- the coarse ring "
            "(docs/extended-occluder-architecture.md)",
            "mesh.latitude_override / longitude_override -- where the core "
            "sits on the body",
            "radiation.view_factor_file -- point at an existing view-factor "
            "matrix rather than rebuilding one",
        ),
    ),
    Preset(
        key="europa_diurnal",
        title="Europa diurnal",
        summary=(
            "Europa over one synodic period with its built-in albedo map. "
            "Radiation exchange off; turn it on for a specific feature."),
        sets={
            "simulation.body": "Europa",
            "simulation.duration_days": 3.551181,
            "simulation.timesteps": 144,
            "mesh.resolution": 6,
            # CLEARED, not merely unset. The engine takes mesh.import_file
            # whenever it is non-empty and reaches the icosphere branch only
            # when it is empty (ptm_thermal_sim.cpp:15087), so applying the
            # crater card and then this one ran the LUNAR CRATER CORE MESH
            # with simulation.body: Europa, and mesh.resolution: 6 was inert.
            # Reproduced.
            "mesh.import_file": "",
            "mesh.occluder_file": "",
            "mesh.latitude_override": -999.0,
            "mesh.longitude_override": -999.0,
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "builtin",
        },
        clears=BUILTIN_BODY_CLEARS,
        beaming="off",
        beaming_note=(
            "The Smith roughness parameterization is calibrated on LUNAR "
            "regolith (w = 32.5 deg, Jhoti et al. 2024) and "
            "docs/radiation-exchange.md declines to carry it to icy bodies. "
            "The control refuses this body regardless of facet scale."),
    ),
    Preset(
        key="generic_body",
        title="Asteroid or other custom body",
        summary=(
            "A generic body driven by HORIZONS orbital elements and spin. Use "
            "the HORIZONS page to fetch one, then this preset to set the rest "
            "of the run up around it."),
        sets={
            # A placeholder, immediately overwritten by the card's own mirror
            # of generic_body.name (GenericBodyCard._mirror_body_name) whenever
            # there is a name to mirror. It is here so that the route defaults
            # alone never leave the PREVIOUS card's body name in place: opening
            # Mercury and then this card used to emit
            # `simulation.body: Mercury` beside a generic_body: section.
            "simulation.body": "Custom",
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "none",
            "illumination.include_primary_reflection": False,
            # generic_body.ephemeris_source is NOT set here. _config_for_yaml
            # emits the generic_body section only once a body is actually
            # active (config_file, name, or a positive semi_major_axis_au), and
            # this card does not make one active -- so reporting it as set was
            # a claim about a key that would be silently dropped, which breaks
            # this module's first rule.
        },
        clears=GENERIC_BODY_CLEARS,
        beaming="off",
        # NOT needs_advanced: nothing this card names sits behind an Advanced
        # SECTION. generic_body.* lives on the Body page, which is gated on a
        # custom body actually being active -- a relevance gate, not a tier --
        # and this preset does not make one active. Claiming "the sections
        # holding the fields above have been opened" was measurably false:
        # advanced on, Body still hidden, generic_body_active False.
        needs=(
            "a custom body -- use the HORIZONS page to search for one and "
            "cache its body_config.yaml. The Body page appears once a custom "
            "body is active; until then there is nothing for it to describe, "
            "and nothing under generic_body: can be written at all.",
            "generic_body.shape -- a radius, semi-axes, or a shape mesh, on "
            "the Body page",
        ),
        beaming_note=(
            "Beaming is offered for lunar bodies only: w = 32.5 deg is a lunar "
            "regolith calibration and there is no measured value for a small "
            "body here."),
    ),
    Preset(
        key="reproduce_baseline",
        title="Reproduce a published baseline",
        summary=(
            "Load the configuration a finished run actually used, from that "
            "run's own config_used.yaml, and report the binary it ran on. The "
            "useful pair to thermal.spinup_seed: legacy.\n"
            "This preset does not guess at values: point it at the run "
            "directory and it reads them.\n"
            "The ENVIRONMENT is the one thing it may not be able to tell you. "
            "config_used.yaml is a byte copy of the config that was passed in, "
            "and provenance.txt records no PTM_* variables at all -- so if that "
            "config carried no engine_env: block (a CLI or campaign launch "
            "usually does not), the baseline's environment is UNKNOWN and this "
            "run will use the GUI's current set instead. The card says which "
            "case you are in."),
        sets={},
        # "preserve", not "off": this card loads a whole configuration and must
        # not overwrite any part of it. With beaming="off" the shared branch
        # forced radiation.beaming False and rms_slope_deg 0 -- silently losing
        # the roughness of a beaming baseline, while its own note said nothing
        # here overrides it. Reproduced through TaskCardsTab.apply_preset.
        beaming="preserve",
        needs=(
            "a completed run directory containing config_used.yaml and "
            "provenance.txt",
        ),
        beaming_note=(
            "Whatever the baseline recorded is what is loaded, including its "
            "beaming setting. Nothing here overrides it."),
    ),
]

PRESETS_BY_KEY = {p.key: p for p in PRESETS}


def apply_preset(config_manager, preset, project_root=None):
    """Apply `preset` to `config_manager` and report exactly what changed.

    Never raises for a mesh it cannot read: the facet-scale rule refuses in
    that direction, which is the safe one.
    """
    result = PresetResult()
    # CLEAR BEFORE SET. A cleared section is reset from DEFAULT_CONFIG, so
    # clearing after `sets` would wipe any key this preset had just written
    # into that section.
    for path in preset.clears:
        removed = config_manager.reset_to_default(path)
        if removed:
            result.cleared.update(removed)
    # A generic body's own shape model goes back immediately. See
    # GENERIC_BODY_CLEARS: the clear above removed the PREVIOUS body's mesh,
    # and this puts THIS body's back, so the two together mean "the mesh is
    # whatever generic_body.shape says" rather than "no mesh".
    if "mesh.import_file" in preset.clears:
        restored = mirror_shape_mesh(config_manager)
        for key, value in restored.items():
            result.applied[key] = value
            result.cleared.pop(key, None)
    for key, value in preset.sets.items():
        config_manager.set_value(key, value)
        result.applied[key] = value

    # --- the beaming rule, evaluated on the config this preset just built ---
    if preset.beaming == "on" and not str(
            config_manager.get_value("mesh.import_file", "") or "").strip():
        # NO MESH YET, so there is nothing to measure. facet_scale falls back
        # to the global icosphere, which on a Moon ico5 reports 43 km and sails
        # through the rule -- so the card turned beaming ON from the facet
        # scale of a mesh it was simultaneously telling the user to go and
        # import. The user then loads the real core mesh, and
        # _update_beaming_gating deliberately does NOT turn an inherited
        # setting off, so the config carries beaming at a scale the rule would
        # have refused. Reproduced: "Beaming ON at w = 32.5 deg. facet scale
        # 43.04 km (20,480 faces, icosphere ...)".
        #
        # Ordering defeats the guarantee, so refuse on ordering.
        config_manager.set_value("radiation.beaming", False)
        config_manager.set_value("radiation.rms_slope_deg", 0.0)
        result.applied["radiation.beaming"] = False
        result.applied["radiation.rms_slope_deg"] = 0.0
        result.refused.append((
            "radiation.beaming",
            "no core mesh is loaded yet, so the facet-scale rule has nothing "
            "to measure -- and measuring the default icosphere instead would "
            "pass a rule written about the mesh you are about to import. "
            "Import the core mesh, then tick Enable Beaming on the Radiation "
            "tab, where the rule measures THAT mesh and refuses if it is at "
            "the roughness scale."))
    elif preset.beaming == "on":
        from gui_core.facet_scale import beaming_policy
        try:
            policy = beaming_policy(config_manager.get_config(), project_root)
        except Exception as exc:                       # never break the apply
            policy = None
            reason = (f"the facet-scale check could not run ({exc}), so "
                      f"beaming is not set. Refusing is the safe direction.")
        else:
            reason = policy.reason
        if policy is not None and policy.allowed and not policy.needs_acknowledgment:
            config_manager.set_value("radiation.beaming", True)
            config_manager.set_value("radiation.rms_slope_deg",
                                     LUNAR_RMS_SLOPE_DEG)
            result.applied["radiation.beaming"] = True
            result.applied["radiation.rms_slope_deg"] = LUNAR_RMS_SLOPE_DEG
            result.notes.append(
                f"Beaming ON at w = {LUNAR_RMS_SLOPE_DEG} deg. {reason}")
        elif policy is not None and policy.needs_acknowledgment:
            config_manager.set_value("radiation.beaming", False)
            result.applied["radiation.beaming"] = False
            result.refused.append((
                "radiation.beaming",
                f"{reason} A preset will not tick a switch that needs an "
                f"explicit acknowledgment -- set it on the Radiation tab, "
                f"where the acknowledgment is."))
        else:
            config_manager.set_value("radiation.beaming", False)
            config_manager.set_value("radiation.rms_slope_deg", 0.0)
            result.applied["radiation.beaming"] = False
            result.applied["radiation.rms_slope_deg"] = 0.0
            result.refused.append(("radiation.beaming", reason))
    elif preset.beaming == "preserve":
        if preset.beaming_note:
            result.notes.append(preset.beaming_note)
    elif preset.beaming in ("off", "n/a"):
        # Explicit, not assumed. `beaming: true` with rms_slope_deg == 0 is a
        # startup FATAL, so a preset that leaves a stale slope behind hands the
        # user a config the engine refuses.
        config_manager.set_value("radiation.beaming", False)
        config_manager.set_value("radiation.rms_slope_deg", 0.0)
        result.applied["radiation.beaming"] = False
        result.applied["radiation.rms_slope_deg"] = 0.0
        if preset.beaming_note:
            result.notes.append(preset.beaming_note)

    for need in preset.needs:
        result.notes.append(f"Still needed: {need}")
    return result


def read_baseline(run_dir):
    """(config_dict, provenance_dict, path) for a finished run directory.

    Raises FileNotFoundError naming what is missing. The point of reading the
    run's OWN record rather than carrying a table of remembered baselines is
    that the record cannot be out of date and this module cannot be wrong
    about it.
    """
    import yaml
    from pathlib import Path

    root = Path(run_dir)
    config_path = root / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} does not exist. 'Reproduce a published baseline' "
            f"reads the configuration a run ACTUALLY used, which the engine "
            f"writes into every output directory as config_used.yaml. Pick "
            f"the run's output directory, not its project file.")
    config = yaml.safe_load(config_path.read_text()) or {}

    provenance = {}
    prov_path = root / "provenance.txt"
    if prov_path.exists():
        for line in prov_path.read_text().splitlines():
            if "=" in line:
                name, _, value = line.partition("=")
                provenance[name.strip()] = value.strip()
    return config, provenance, config_path


def compare_binary(provenance, current_md5):
    """One line about whether you are about to run the baseline's binary.

    A reproduction on a different binary is not a reproduction, and the md5 is
    the only thing that settles it -- a quoted commit hash has been wrong here
    before.
    """
    recorded = provenance.get("engine_binary_md5", "")
    sha = provenance.get("engine_git_sha", "?")
    # run_provenance.cpp writes the literal string "unknown" when it could not
    # resolve its own executable path. Treating that as a hash produced
    # "DIFFERENT BINARY. The baseline ran on md5 unknown" -- alarming, and
    # wrong: the truth is that the baseline recorded no md5 at all.
    if recorded.strip().lower() in ("", "unknown", "n/a", "none"):
        recorded = ""
    if not recorded:
        return ("This run directory records no engine md5, so whether your "
                "binary matches the baseline's cannot be checked here.")
    if not current_md5:
        return (f"The baseline ran on md5 {recorded} (git {sha}). The current "
                f"binary could not be fingerprinted, so they cannot be "
                f"compared.")
    if recorded == current_md5:
        return f"Your binary IS the baseline's: md5 {recorded} (git {sha})."
    return (f"DIFFERENT BINARY. The baseline ran on md5 {recorded} (git "
            f"{sha}); yours is {current_md5}. A reproduction on a different "
            f"binary is not a reproduction -- the configuration below is the "
            f"baseline's, the engine is not.")
