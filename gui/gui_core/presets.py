"""Reachable configuration presets (item 2.3 / DP-5).

``ConfigManager.PRESETS`` used to hold five of these and nothing ever called
them -- ``grep -rn load_preset gui/`` returned the definition and nothing else
-- while ``gui/README.md`` advertised the feature. Stage 1 deleted the dead
table rather than pretend it worked (GUI-PRESET-1); this is the replacement,
built from §4.4 of ``docs/design/gui-update-plan.md``.

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


@dataclass
class PresetResult:
    """What applying a preset actually did."""
    applied: dict = field(default_factory=dict)   # dotted key -> value written
    refused: list = field(default_factory=list)   # (dotted key, reason)
    notes: list = field(default_factory=list)     # things the user must supply

    def summary_lines(self):
        out = [f"Set {len(self.applied)} key(s):"]
        for key in sorted(self.applied):
            out.append(f"    {key} = {self.applied[key]!r}")
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
    #: "on" (subject to the scale rule), "off", or "n/a" (radiation is off).
    beaming: str = "off"
    #: Fields the user still has to fill in for this to be a runnable config.
    needs: tuple = ()
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
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "builtin",
        },
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
        beaming="on",
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
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "builtin",
        },
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
            "thermal.run_thermal_mesh": True,
            "radiation.enabled": False,
            "illumination.albedo_source": "none",
            "illumination.include_primary_reflection": False,
            # A shaped body carries its own thermophysical values through
            # generic_body.surface; the built-in Moon/Europa numbers do not
            # apply to it.
            "generic_body.ephemeris_source": "auto",
        },
        beaming="off",
        needs=(
            "generic_body.name and generic_body.config_file -- fetch them from "
            "the HORIZONS page",
            "generic_body.shape -- a radius, semi-axes, or a shape mesh",
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
            "run's own config_used.yaml, and report the binary it ran on. This "
            "is the useful pair to thermal.spinup_seed: legacy -- and it would "
            "have made PTM-276 impossible to hit by accident, because the "
            "environment comes from the baseline's record instead of from "
            "whatever the GUI injects today.\n"
            "This preset does not guess at values: point it at the run "
            "directory and it reads them."),
        sets={},
        beaming="off",
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
    for key, value in preset.sets.items():
        config_manager.set_value(key, value)
        result.applied[key] = value

    # --- the beaming rule, evaluated on the config this preset just built ---
    if preset.beaming == "on":
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
