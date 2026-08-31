"""Display labels and inline help for the PTM3D configuration panel.

THIS FILE IS THE SINGLE SOURCE for both. ``config_panel.py`` builds widgets;
it does not carry their prose. A control whose key appears in ``TOOLTIPS``
takes its help text from here even if its call site also passes one, so the
two can never drift apart, and ``LABELS`` overrides the call site's label the
same way. ``src/python/tests/test_gui_tooltip_coverage.py`` pins both.

Each key is a configuration path (for example ``simulation.body``). Keys
beginning ``ui.`` are controls that have no configuration key of their own.

HOW TO WRITE ONE. The audience is a planetary scientist who has not read the
source. Answer three questions, in this order, and stop:

1. What does this physically do?
2. What is the recommended value, and why is it recommended?
3. When would I change it?

For an enum, give each choice its own line. Name the YAML key at the end, so a
label on screen and a key in a file can always be matched up.

Every physical claim here must be checkable against the engine or the docs --
a wrong tooltip is worse than no tooltip, because it is believed. No ticket
numbers, no internal shorthand, and American English throughout.
"""

#: Plain-language display labels, keyed by configuration path.
#:
#: The YAML key is NEVER renamed -- the engine reads it. This table only
#: changes what the panel *shows*, and every entry names its own key in the
#: matching tooltip so a label and a config file can always be matched up.
#: A key absent from here keeps whatever label its call site passes.
LABELS = {
    # -- the equilibration / initial-condition group ------------------------
    "thermal.spinup_seed": "Starting temperature profile:",
    "thermal.seed_anchor": "Surface anchor for that profile:",
    "thermal.analytic_seed_cycles": "Clear-sky pre-conditioning (cycles):",
    "thermal.seed_anchor_refuse_two_timescale":
        "Refuse the anchor on daily + seasonal faces",
    "thermal.equilibration_target_K": "Spin-up accuracy target (K):",
    "thermal.equilibration_stop_on_certificate":
        "Let spin-up stop early once that accuracy is reached",
    "thermal.equilibration_cycles": "Spin-up cycle count (override):",
    "thermal.eq_indirect_passes": "Neighbor-infrared refinement passes:",
    "thermal.eq_indirect_source": "Neighbor infrared during spin-up:",
    "thermal.eq_indirect_flux_file": "Neighbor-infrared file (optional):",
    # -- radiation ----------------------------------------------------------
    "radiation.beaming": "Beam infrared from sub-facet roughness",
    "radiation.rms_slope_deg": "Sub-facet RMS slope (deg):",
    "radiation.min_shadow_fraction":
        "Shadow-fraction threshold (unused by default):",
    "radiation.nearfield_quadrature_order":
        "Near-field accuracy (quadrature order):",
    "radiation.use_monte_carlo_vf": "Build view factors by ray casting",
    "radiation.mc_rays_per_face": "Rays cast per face:",
    "radiation.coupling_mode": "Coupling stop criterion:",
    # -- mesh ---------------------------------------------------------------
    "mesh.occluder_file": "Shadow-only surrounding mesh:",
    # -- run state ----------------------------------------------------------
    "thermal.initial_state_file": "Warm start from state file:",
    "thermal.final_state_file": "Save final state to:",
    # -- ice stability ------------------------------------------------------
    "thermal.ice_stability.flux_threshold_kgm2s":
        "Ice survives below this loss rate (kg m^-2 s^-1):",
    "thermal.ice_stability.grain_hop_length_um":
        "Pore path length (micrometers):",
    "thermal.ice_stability.n_depth_subsamples":
        "Depth subsamples between nodes:",
    # -- depth grid ---------------------------------------------------------
    "thermal.thermal_n": "Layer growth factor n:",
    # -- flux cubes ---------------------------------------------------------
    "flux_cubes.enabled": "Record irradiance at virtual measurement cubes",
    "flux_cubes.mode": "Cube placement:",
    "flux_cubes.spacing_m": "Distance between cubes (m):",
    "flux_cubes.radius_m": "Grid radius around the mesh center (m):",
    "flux_cubes.cache_file": "Cube layout file:",
    "flux_cubes.write_binary": "Write the single binary flux file",
    "flux_cubes.write_csv": "Write one spreadsheet file per cube",
    "flux_cubes.write_csv_max_cubes": "Spreadsheet files, at most:",
    "flux_cubes.shadow_probe_rays": "Sun-visibility test:",
}

TOOLTIPS = {
    # =========================================================================
    # SIMULATION TAB
    # =========================================================================
    "simulation.body": (
        "Which body is being modeled. The choice sets the default orbit, spin, "
        "surface properties and ephemeris kernels; a `[H] ...` entry is one "
        "fetched from JPL Horizons.\n"
        "Pick the body first -- almost every other default on this panel is "
        "derived from it.\n"
        "YAML key: simulation.body"
    ),
    "simulation.start_time": (
        "The epoch the simulated time series begins at, in UTC "
        "(for example 2030-04-10T00:00:00). It fixes the Sun's position "
        "through the ephemeris, so it sets the season and the local time of "
        "the first output frame.\n"
        "Any epoch inside the loaded ephemeris coverage will do for a diurnal "
        "study. Choose deliberately for a seasonal one.\n"
        "YAML key: simulation.start_time"
    ),
    "simulation.duration_days": (
        "How much time the recorded output covers, counted after spin-up has "
        "finished. The dropdown only changes the units on display; the value "
        "is always stored in Earth days.\n"
        "One solar day of the selected body is the usual choice, and the "
        "default when you pick a body: it is exactly one diurnal cycle.\n"
        "Lengthen it for a seasonal run, or to average over several cycles.\n"
        "YAML key: simulation.duration_days"
    ),
    "simulation.end_time": (
        "The epoch the run finishes at, derived from the start time and the "
        "duration. Editing it adjusts the duration and leaves the start time "
        "alone, so you can specify a window by its end instead of its length."
    ),
    "simulation.timesteps": (
        "How many frames the output window is split into -- the sampling "
        "cadence of everything written to disk.\n"
        "Aim for at least about 48 per diurnal cycle: the surface temperature "
        "peak is sharp, and a coarser cadence clips it. The default of 72 is "
        "comfortable for one cycle.\n"
        "Raise it for sharp features such as terminator crossings or "
        "eclipses; every frame costs an illumination solve and output volume.\n"
        "YAML key: simulation.timesteps"
    ),

    # =========================================================================
    # MESH TAB
    # =========================================================================
    "mesh.resolution": (
        "Subdivision level of the built-in global icosphere. Each level "
        "splits every triangle into four, so the mesh has 20 x 4^level "
        "faces: level 4 = 5,120, level 5 = 20,480, level 6 = 81,920, "
        "level 7 = 327,680, level 8 = 1,310,720.\n"
        "Level 5 is the default and is enough for a global survey; level 6 is "
        "the usual choice for published global maps.\n"
        "Ignored entirely when a mesh is imported below.\n"
        "YAML key: mesh.resolution"
    ),
    "mesh.use_topographic": (
        "Use a pre-built topographic global mesh for this body instead of a "
        "smooth sphere, so real relief casts real shadows.\n"
        "Off by default, because it needs the matching elevation dataset on "
        "disk. Turn it on for any study where topography matters and leave it "
        "off for a smooth-sphere reference run.\n"
        "YAML key: mesh.use_topographic"
    ),
    "mesh.topo_directory": (
        "Folder holding the pre-built topographic meshes, used only in Custom "
        "mode. Leave the source set to Built-in and this is resolved for you "
        "from the shipped data directory.\n"
        "YAML key: mesh.topo_directory"
    ),
    "mesh.import_file": (
        "Your own mesh, in PLY, OBJ or VTK form, used instead of the built-in "
        "sphere. It should be a triangle mesh; a closed body mesh gives "
        "correct self-shadowing everywhere, while an open terrain patch is "
        "shadowed only by what it contains, plus any occluder mesh below.\n"
        "Leave blank to use the icosphere.\n"
        "YAML key: mesh.import_file"
    ),
    "mesh.scale_factor": (
        "Multiplier applied to the imported mesh's coordinates. The engine "
        "works in the mesh's own units, and those units decide the physical "
        "facet size that several roughness and view-factor rules test.\n"
        "1.0 is right when the file is already in the units you want. Small "
        "body shape models are usually in kilometers and lunar or Europan "
        "elevation meshes in meters, so check before trusting the default -- "
        "a wrong scale moves every length in the run by that factor.\n"
        "YAML key: mesh.scale_factor"
    ),
    "mesh.occluder_file": (
        "A coarse, shadow-only mesh wrapped around a high-resolution core "
        "mesh. It joins the shadow-ray scene so terrain outside the core can "
        "block the Sun, but it is never thermally solved, never coupled into "
        "the radiosity exchange and never written to the output.\n"
        "Leave blank for a global mesh. Set it for any site or regional run "
        "where relief beyond the edge of the core casts shadows into it -- "
        "otherwise the core behaves as if it sat on a flat plain.\n"
        "It must be a separate file from the core: a ring mesh has no faces "
        "inside the core radius. See docs/extended-occluder-architecture.md.\n"
        "YAML key: mesh.occluder_file"
    ),
    "mesh.latitude_override": (
        "Latitude to place the imported mesh at, in degrees. The engine "
        "rotates the whole mesh so its own frame origin lands there.\n"
        "-999 leaves the mesh exactly where its file put it, which is right "
        "whenever the file already carries real body-fixed coordinates.\n"
        "This field is also the switch for the pair: the rotation happens "
        "only when a latitude is given, so a longitude on its own does "
        "nothing at all.\n"
        "YAML key: mesh.latitude_override"
    ),
    "mesh.longitude_override": (
        "Longitude to place the imported mesh at, in degrees.\n"
        "-999 leaves the mesh in its own frame.\n"
        "READ ONLY WHEN A LATITUDE OVERRIDE IS ALSO SET. The engine gates the "
        "entire rotation on latitude, so a longitude given on its own is "
        "ignored without a word -- which is why this field stays disabled "
        "until the latitude above has a value.\n"
        "YAML key: mesh.longitude_override"
    ),

    # =========================================================================
    # THERMAL TAB
    # =========================================================================
    "thermal.run_thermal_mesh": (
        "Solve the heat equation on the mesh. With this off the run computes "
        "illumination only -- shadows, incidence angles and absorbed flux -- "
        "which is much faster and is what you want for a visibility or "
        "lighting study.\n"
        "Turn it on for any temperature result.\n"
        "YAML key: thermal.run_thermal_mesh"
    ),
    "thermal.threads": (
        "How many processor cores the solver may use.\n"
        "0 (the default) means use every core the machine reports, which is "
        "almost always what you want.\n"
        "Set a smaller number to leave headroom for other work.\n"
        "YAML key: thermal.threads"
    ),
    "thermal.equilibration_orbits": (
        "How long the model is run before recording starts, so the "
        "subsurface can forget its starting temperatures and settle into a "
        "repeating cycle. Nothing from this period is written out. The "
        "dropdown changes only the units on display.\n"
        "One solar day of the body is the default and is enough for the "
        "near-surface layers. The deep column is much slower: high thermal "
        "inertia, an insulating surface layer, or any result that depends on "
        "temperatures at depth needs several cycles or more.\n"
        "If the deep temperatures still drift between the start and end of "
        "the output window, this is the number to raise.\n"
        "YAML key: thermal.equilibration_orbits"
    ),
    "thermal.compute_statistics": (
        "Reduce the time series into per-face minimum, maximum and mean "
        "temperature maps, written next to the frames.\n"
        "On by default and cheap. These files are what most figures and most "
        "of the analysis tooling read, so leave it on unless you only want "
        "raw frames.\n"
        "YAML key: thermal.compute_statistics"
    ),
    "thermal.enable_shadow_flux": (
        "Give a face that is in shadow, but whose Sun is still above the "
        "local horizon, an analytic estimate of the sunlight scattered down "
        "onto it by the sunlit walls of an idealized bowl crater. The bowl's "
        "shape comes from the ratio below, not from the mesh.\n"
        "Off by default. It is a stand-in for a crater the mesh does not "
        "resolve, so it belongs on a smooth or coarse mesh only.\n"
        "Leave it off whenever radiation exchange is on and the mesh really "
        "does resolve the crater: the two describe the same energy, and "
        "running both counts it twice.\n"
        "YAML key: thermal.enable_shadow_flux"
    ),
    "thermal.crater_dd_ratio": (
        "Shape of that idealized bowl, given as its DIAMETER divided by its "
        "depth. 8 (the default) is a bowl eight times wider than it is deep, "
        "a typical simple crater.\n"
        "Smaller values mean a deeper, more enclosed bowl, which traps and "
        "scatters more light; larger values mean a wide, shallow one that "
        "traps little. Above 10 the engine switches to its shallow-crater "
        "approximation.\n"
        "Inert unless the scattered shadow flux above is on.\n"
        "YAML key: thermal.crater_dd_ratio"
    ),
    "thermal.geothermal_heat_flow": (
        "Heat arriving from the interior at the bottom of the modeled "
        "column, in W/m^2. It sets the floor a face can cool to: a surface "
        "that never sees the Sun settles at (Qb / (emissivity x sigma))^(1/4) "
        "-- about 24 K on the Moon.\n"
        "-1 (the default) uses the body's own value: about 0.018 W/m^2 for "
        "the Moon, 0.03 for Europa.\n"
        "Set it explicitly for a body with unusual internal heating, or to "
        "test how sensitive a cold trap is to that floor. This value is "
        "applied on every body, including the built-in ones.\n"
        "YAML key: thermal.geothermal_heat_flow"
    ),
    "thermal.use_gpu": (
        "Run the heat-equation solver on the graphics card instead of the "
        "processor. The physics is the same; this is a speed choice, and it "
        "wins on large meshes.\n"
        "Off by default. Turn it on for meshes of roughly 100,000 faces and "
        "up. It is also required for the ice-stability product below, which "
        "only exists on this path.\n"
        "YAML key: thermal.use_gpu"
    ),
    "thermal.integration_days": (
        "Length of the window the run's minimum, mean and maximum are "
        "collected over, in days. It is measured from the start of the output "
        "window, so a window shorter than the run reports the first part of "
        "it.\n"
        "On a body card this is the run length itself, given as a count of "
        "diurnal or orbital periods and always a real duration; the output "
        "window is set to match, so the run integrates exactly what it "
        "simulates.\n"
        "-1, which this advanced view still accepts, means integrate exactly "
        "the simulation duration.\n"
        "YAML key: thermal.integration_days"
    ),
    "thermal.integration_steps": (
        "Number of internal solver steps across the integration window, when "
        "it should differ from the number of output frames.\n"
        "-1 (the default) lets the engine choose from the thermal time "
        "constant. Override only to reproduce a specific historical run.\n"
        "YAML key: thermal.integration_steps"
    ),
    "thermal.thermal_inertia_map": (
        "A per-face map that lets the surface vary across the mesh instead of "
        "taking one value everywhere: a raw binary file of one 32-bit float "
        "per face, in mesh face order, and it must have exactly as many "
        "values as the mesh has faces.\n"
        "Read only when Map Mode is 'Custom file'. WHICH QUANTITY IT CARRIES "
        "is set by Map Content above, not by this field -- the same path is "
        "saved as h_parameter_map when that says H-parameter, and as "
        "thermal_inertia_map when it says thermal inertia. Only ever one of "
        "the two.\n"
        "Leave blank to use the uniform properties from the Properties tab.\n"
        "YAML keys: thermal.h_parameter_map / thermal.thermal_inertia_map"
    ),
    "thermal.property_map_mode": (
        "Where the per-face surface-property map comes from -- the map that "
        "lets the surface vary across the mesh instead of taking one value "
        "everywhere.\n"
        "None (uniform body H): every face uses the body's own value. The "
        "default, and the right choice unless you have a map.\n"
        "Bundled icosphere default for body: use the map that ships with "
        "PTM3D, found automatically under the property-maps directory for "
        "this body.\n"
        "Custom file: load the file named below.\n"
        "YAML keys: thermal.use_default_property_map, "
        "thermal.h_parameter_map / thermal.thermal_inertia_map"
    ),
    "thermal.property_map_kind": (
        "Which quantity the per-face map file actually holds. This selector, "
        "not the file name, decides how the values are read.\n"
        "H-parameter (m): each value is that face's own e-folding depth, the "
        "scale over which density and conductivity climb toward their deep "
        "values.\n"
        "Thermal inertia (J m^-2 K^-1 s^-1/2): each value is that face's "
        "thermal inertia at 273 K. The loader converts it into an equivalent "
        "H per face, so both kinds end up driving the same depth model -- "
        "which is why the two are alternatives and never both set.\n"
        "YAML keys: thermal.h_parameter_map / thermal.thermal_inertia_map"
    ),
    "thermal.property_maps_directory": (
        "Folder searched for the bundled per-face property maps. The default "
        "points at the shipped data directory; change it only if you keep "
        "these files somewhere else.\n"
        "YAML key: thermal.property_maps_directory"
    ),

    # -- Equilibration convergence -------------------------------------------
    "thermal.equilibration_target_K": (
        "How close to a perfectly repeating cycle the spin-up is asked to "
        "get, in kelvin. The engine scores its own convergence estimate "
        "against this bar and reports the result.\n"
        "0.1 K is the default and is a sensible bar for surface temperatures. "
        "Loosen it to around 1 K for a quick look; tighten it for deep-column "
        "or ice-stability work, where the slow deep layers are the answer.\n"
        "On its own this only changes what is reported. It shortens a run "
        "only when 'let spin-up stop early' below is also on.\n"
        "YAML key: thermal.equilibration_target_K"
    ),
    "thermal.equilibration_stop_on_certificate": (
        "Let spin-up end as soon as the convergence estimate reaches the "
        "target accuracy above, instead of always running the full "
        "equilibration period.\n"
        "Off by default, and that is the right choice for anything you will "
        "compare: a fixed spin-up budget gives every run in a campaign the "
        "same treatment, and the convergence estimate is an estimate, not a "
        "proof.\n"
        "Turn it on for exploratory single runs where spin-up dominates the "
        "wall clock.\n"
        "YAML key: thermal.equilibration_stop_on_certificate"
    ),
    "thermal.equilibration_cycles": (
        "An exact spin-up cycle count, overriding the equilibration period "
        "set higher up this tab.\n"
        "-1 (the default) means follow that field, in whichever units you "
        "chose there.\n"
        "Set a positive number only when a specific cycle count matters, such "
        "as matching a published run.\n"
        "YAML key: thermal.equilibration_cycles"
    ),
    "thermal.eq_indirect_passes": (
        "How many times the spin-up phase re-estimates the infrared each face "
        "receives from its neighbors and repeats the spin-up solve with it. "
        "The first pass estimates that flux from temperatures that "
        "equilibrated with no neighbor infrared at all, so it comes out low "
        "-- measured 11.7 percent low on the shipped 80N crater. Later "
        "passes re-estimate it from the temperatures the previous pass "
        "produced.\n"
        "1 is the default and is the right value for most work. On that same "
        "crater a second pass bought 0.28 K of deep column for about a "
        "quarter more wall clock, while switching the source below to "
        "'converged' at one pass bought 3.9 K. Spend on the source first.\n"
        "The engine refuses 0; to switch the term off, use the source below.\n"
        "YAML key: thermal.eq_indirect_passes"
    ),
    "thermal.eq_indirect_source": (
        "Where the spin-up phase gets the infrared each face receives from "
        "its neighbors. Only matters when radiation exchange is on, and it is "
        "what keeps a shadowed face warmed by its own crater walls from "
        "spinning up as if those walls were not there.\n"
        "auto (default): estimated from a radiosity sweep taken at the start "
        "of the coupling loop. Cheap, but the emitters in that sweep do not "
        "yet carry any neighbor infrared themselves, so the estimate stays "
        "13.3 percent low on that crater NO MATTER HOW MANY PASSES you add. "
        "That is a different limitation from the first-pass bias on the "
        "line above: extra passes cure that one, and cannot cure this "
        "one.\n"
        "converged: the same machinery, read after the coupling loop has "
        "converged. This is the only setting that removes that low bias. It "
        "costs roughly twice the wall clock on the shipped 80N crater and is "
        "worth it for deep-column, cold-trap and ice-stability work.\n"
        "off: spin-up sees no neighbor infrared at all, so a face whose only "
        "heat source is its neighbors settles to the bare geothermal floor. "
        "Use it only to reproduce a run published under that behavior.\n"
        "Spelled exactly: the engine refuses 'Auto' at startup.\n"
        "YAML key: thermal.eq_indirect_source"
    ),
    "thermal.eq_indirect_flux_file": (
        "A file of per-face mean neighbor-infrared flux written by an earlier "
        "run, used to feed the spin-up phase directly.\n"
        "Leave blank. When set it overrides the source setting above "
        "entirely; it exists for the manual two-run workflow that the "
        "internal estimate replaced.\n"
        "YAML key: thermal.eq_indirect_flux_file"
    ),

    # -- Initial condition ----------------------------------------------------
    "thermal.spinup_seed": (
        "How each face's temperature column is filled in before spin-up "
        "starts.\n"
        "physical (recommended): every face gets its own conductive "
        "steady-state profile, computed from that face's own mean absorbed "
        "sunlight -- surface set by radiative balance, then temperature "
        "rising with depth at the rate the interior heat flow requires. A "
        "permanently shadowed column therefore starts on the geothermal "
        "floor, instead of the 42 to 47 K above it that the older seeds "
        "produced and a short spin-up cannot shed.\n"
        "legacy: reproduces the pre-2026 seeds, which differed from one "
        "solver path to another, together with the clear-sky pre-conditioner. "
        "Keep it only for reproducing an older run exactly.\n"
        "A correct solver forgets its starting point, so on a well "
        "equilibrated run this changes nothing. It matters most where "
        "spin-up is short or the column is slow -- polar craters, cold traps "
        "and deep temperatures.\n"
        "YAML key: thermal.spinup_seed"
    ),
    "thermal.seed_anchor": (
        "Which surface temperature that starting profile is hung from. Read "
        "only when the starting profile above is 'physical'.\n"
        "lumped (recommended): the cycle-mean surface temperature, obtained "
        "by putting a simple heat-storing substrate under the face and "
        "matching its response to that face's own sunlight variation. The "
        "conductive profile describes the cycle mean, so this is the "
        "temperature it should start from.\n"
        "warm: the fourth-power mean, ((mean absorbed flux + Qb) / "
        "(emissivity x sigma))^(1/4). It is exact for a face whose "
        "temperature barely varies, but for a face that swings through a day "
        "it is always too warm -- about 45 K too warm on the sunlit faces of "
        "the shipped 80N lunar crater. It reproduces runs made before "
        "mid-August 2026 exactly.\n"
        "Change it only to reproduce such a run.\n"
        "YAML key: thermal.seed_anchor"
    ),
    "thermal.analytic_seed_cycles": (
        "Diurnal cycles of simplified clear-sky sunlight integrated before "
        "the real forcing begins. This is a forcing, not a starting "
        "condition: it drives the column with smooth cos(latitude) x "
        "cos(hour angle) insolation and no topography, no shadowing, no "
        "infrared from neighbors and no scattered light, so it overwrites the "
        "per-face profile that was just computed. It exists on the processor "
        "solver only; the graphics-card solver has no equivalent.\n"
        "-1 (recommended) follows the starting profile above, which means it "
        "is off under 'physical'. 0 forces it off.\n"
        "Set a positive count only when reproducing an older processor run "
        "that depended on it. It does not count toward the equilibration "
        "period.\n"
        "YAML key: thermal.analytic_seed_cycles"
    ),
    "thermal.seed_anchor_refuse_two_timescale": (
        "What to do about a face whose sunlight varies on two very different "
        "periods at once -- the daily cycle plus a slower seasonal drift. Any "
        "run spanning several cycles has some: over two lunations the "
        "sub-solar latitude moves, and that drift is a second timescale. The "
        "lumped anchor fits one heat-storage number to one frequency, so on "
        "such a face it is covering two regimes at once.\n"
        "Leave it OFF (recommended). Refusing does not skip the face -- it "
        "falls back to the 'warm' anchor, which is the biased estimator. "
        "Measured on the shipped 80N crater over two lunations, this fires on "
        "158 of 12,800 faces; 111 of those are dark and flat, where the warm "
        "anchor is already right, but on the 47 that genuinely swing, "
        "refusing costs about 70 K of error at the median and 84 K at worst, "
        "where accepting the anchor costs 3.7 K and 13.4 K.\n"
        "Turn it on only if you want the strict older behavior. Detection and "
        "reporting happen either way, so the run tells you which faces are "
        "affected regardless.\n"
        "YAML key: thermal.seed_anchor_refuse_two_timescale"
    ),

    # -- Run state ------------------------------------------------------------
    "thermal.initial_state_file": (
        "A file of per-face subsurface temperature profiles to start from, "
        "written by an earlier run. Loading one is a warm start: the run "
        "continues where the other stopped and skips spin-up entirely.\n"
        "Leave blank for a normal run.\n"
        "Use it to continue a long simulation in pieces. The result matches a "
        "single unbroken run only if the pieces line up in time, which is "
        "what the chunked-run script exists to guarantee.\n"
        "YAML key: thermal.initial_state_file"
    ),
    "thermal.final_state_file": (
        "Where to write the per-face subsurface temperature profiles when the "
        "run finishes, so a later run can continue from them.\n"
        "Leave blank unless you intend to continue. The state saved is the "
        "one at the last exported frame, which is what the next piece must "
        "resume from.\n"
        "YAML key: thermal.final_state_file"
    ),

    # -- Ice stability --------------------------------------------------------
    "thermal.compute_ice_stability": (
        "Work out, for every face, the depth below which water ice would "
        "survive over geological time, and write it beside the temperature "
        "output as ice_depth.bin. Following Schorghofer and Taylor (2007), "
        "the engine averages the sublimation rate over the modeled cycle at "
        "every depth node, then finds the depth at which vapor loss through "
        "the overlying regolith falls under the threshold below.\n"
        "Off by default and free when off. Turn it on for cold-trap and "
        "volatile-stability studies.\n"
        "It is produced by the graphics-card thermal solver only: with that "
        "solver off, the run still completes but writes no ice file.\n"
        "YAML key: thermal.compute_ice_stability"
    ),
    "thermal.ice_stability.flux_threshold_kgm2s": (
        "The sublimation loss rate below which ice counts as stable.\n"
        "The default, 3e-14 kg m^-2 s^-1, is roughly one millimeter lost per "
        "billion years -- the usual definition of geologically stable.\n"
        "Raise it for a looser definition over a shorter era.\n"
        "YAML key: thermal.ice_stability.flux_threshold_kgm2s"
    ),
    "thermal.ice_stability.grain_hop_length_um": (
        "The typical distance a water molecule travels through the pore space "
        "between grains, which sets how efficiently vapor escapes from depth. "
        "A larger value means easier escape and therefore a deeper stability "
        "horizon.\n"
        "75 micrometers is the default, a regolith grain scale.\n"
        "Change it with a measured or assumed grain size for the surface you "
        "are modeling.\n"
        "YAML key: thermal.ice_stability.grain_hop_length_um"
    ),
    "thermal.ice_stability.sublimation_enthalpy_kJmol": (
        "Latent heat of sublimation of water ice, used in the vapor-pressure "
        "law that converts temperature into a loss rate.\n"
        "51.058 kJ/mol is the default and is the standard value for water "
        "ice. Change it only when modeling a different volatile.\n"
        "YAML key: thermal.ice_stability.sublimation_enthalpy_kJmol"
    ),
    "thermal.ice_stability.geothermal_gradient_K_per_m": (
        "Temperature rise per meter applied below the deepest modeled layer, "
        "so the stability search can continue past the bottom of the "
        "column.\n"
        "0 (the default) treats everything below the column as isothermal, "
        "which is the conservative choice.\n"
        "Set a real gradient when the stability horizon is deep enough that "
        "interior warming would close it off from below.\n"
        "YAML key: thermal.ice_stability.geothermal_gradient_K_per_m"
    ),
    "thermal.ice_stability.n_depth_subsamples": (
        "How finely to interpolate between depth nodes when locating the "
        "stability horizon. More subsamples give a smoother depth, not a "
        "different physical answer.\n"
        "-1 (the default) lets the engine choose. Raise it only if the depth "
        "map looks stepped at the layer spacing.\n"
        "YAML key: thermal.ice_stability.n_depth_subsamples"
    ),

    # -- Depth grid (read-only) ----------------------------------------------
    "thermal.thermal_b": (
        "Where the bottom of the modeled column sits, measured in diurnal "
        "skin depths -- the depth scale over which the daily temperature wave "
        "dies away.\n"
        "The default of 15 leaves the daily wave thoroughly damped before the "
        "insulating bottom boundary, so that boundary does not distort the "
        "result.\n"
        "Shown here, not offered: deepening the column without also "
        "lengthening spin-up makes the deep answer worse before it gets "
        "better, because the added layers start further from equilibrium. "
        "Edit the YAML directly if you have a reason, and raise the "
        "equilibration period with it.\n"
        "YAML key: thermal.thermal_b"
    ),
    "thermal.thermal_m": (
        "How many layers fill the topmost skin depth -- the vertical "
        "resolution where the daily wave lives and where the surface "
        "temperature is decided.\n"
        "The default is 10. Shown here, not offered.\n"
        "YAML key: thermal.thermal_m"
    ),
    "thermal.thermal_n": (
        "The layer growth factor of the OLD fixed geometric depth grid.\n"
        "NOT USED by a current run. The solver builds an adaptive grid per "
        "profile, and that path never reads this value; only the retired "
        "non-adaptive grid did. It is displayed so that a configuration "
        "carrying a value does not look as though it is doing something.\n"
        "The near-surface resolution that does matter is the nodes-per-skin-"
        "depth figure above.\n"
        "YAML key: thermal.thermal_n"
    ),

    # -- Thermal GPU dispatch -------------------------------------------------
    "thermal.gpu_devices": (
        "How many of the graphics cards found on this machine the thermal "
        "solver may spread its work across.\n"
        "1 (the default) uses the first one; 0 means use every one.\n"
        "Ignored when an explicit device list is given below. This is a speed "
        "setting: the temperatures do not depend on it.\n"
        "YAML key: thermal.gpu_devices"
    ),
    "thermal.gpu_device_ids": (
        "Exactly which graphics cards to use, by index, as a comma-separated "
        "list such as 0,1. Overrides the count above.\n"
        "Leave blank unless you need to keep off a specific card -- for "
        "example one driving the display, or one another job is using.\n"
        "Run ptm_thermal_sim --list-gpu-devices to see the indices.\n"
        "YAML key: thermal.gpu_device_ids"
    ),
    "thermal.gpu_batch_size": (
        "How many faces are handed to the graphics card in one go. Smaller "
        "batches mean shorter individual pieces of work.\n"
        "0 (the default) lets the engine pick, either from its own cap or "
        "from this machine's measured profile.\n"
        "Lowering it is the first thing to try if the display driver resets "
        "the card mid-run: a long single piece of work is what trips that "
        "watchdog. It does not change the answer.\n"
        "YAML key: thermal.gpu_batch_size"
    ),
    "thermal.gpu_inter_batch_pause_ms": (
        "An idle gap inserted between batches, in milliseconds.\n"
        "0 (the default) means no gap.\n"
        "A pause of a few tens of milliseconds lets the graphics driver see "
        "the card as responsive between batches, which is the second "
        "mitigation for a driver reset on this hardware. It costs wall clock "
        "and changes nothing physical.\n"
        "YAML key: thermal.gpu_inter_batch_pause_ms"
    ),
    "thermal.auto_calibrate_gpu": (
        "Let the engine measure this machine's safe dispatch size once, on "
        "the first run that uses the card, and remember it.\n"
        "On by default, and worth leaving on: the measured ceiling is usually "
        "much better than the safe guess.\n"
        "Turning it off makes the engine fall back to deliberately "
        "conservative limits, and it says so in the log. Numerically inert "
        "either way.\n"
        "YAML key: thermal.auto_calibrate_gpu"
    ),

    # =========================================================================
    # PROPERTIES (THERMOPHYSICAL) TAB
    # =========================================================================
    "thermophysical.albedo": (
        "Bond albedo: the fraction of incoming sunlight the surface reflects "
        "over all wavelengths and directions. What is left is absorbed and "
        "drives the temperature.\n"
        "About 0.12 for the Moon, 0.5 for Europa. Untick 'use default' to "
        "override the body's value.\n"
        "YAML key: thermophysical.albedo"
    ),
    # The five properties a BUILT-IN body can override. The suffix matters:
    # on the Moon, Europa or Mercury the engine reads these and ignores the
    # unsuffixed names above, which are honored only for a custom body.
    "thermophysical.albedo_override": (
        "Bond albedo for this run: the fraction of sunlight the surface "
        "reflects at all wavelengths and in all directions. The rest is "
        "absorbed, so lowering it makes the surface hotter.\n"
        "Leave the tick set to use the engine's own value for this body -- "
        "0.12 for the Moon, 0.5 for Europa, 0.07 for Mercury. Untick it to "
        "study a brighter or darker surface than the body's average.\n"
        "YAML key: thermophysical.albedo_override"
    ),
    "thermophysical.emissivity_override": (
        "How efficiently the surface radiates heat away, as a fraction of a "
        "perfect radiator at the same temperature. Lower values trap heat and "
        "raise the daytime peak.\n"
        "Leave the tick set for the engine's value -- 0.95 for the Moon, 0.9 "
        "for Europa. Rock and ice are close to 1, so this is rarely moved far "
        "from it.\n"
        "YAML key: thermophysical.emissivity_override"
    ),
    "thermophysical.ks_override": (
        "How well the loose surface layer conducts heat, in watts per meter "
        "per kelvin. This is the single most important number for how far the "
        "daily temperature swing reaches, and how cold the night gets.\n"
        "Leave the tick set for the engine's value -- 0.00074 for lunar "
        "regolith, 0.0096 for Europa's ice. Raise it for a rockier or more "
        "compacted surface.\n"
        "YAML key: thermophysical.ks_override"
    ),
    "thermophysical.kd_override": (
        "How well the deep, compacted material conducts heat, in watts per "
        "meter per kelvin. It sets the temperature the surface relaxes toward "
        "at depth, below the reach of the daily cycle.\n"
        "Leave the tick set for the engine's value -- 0.0034 for the Moon, "
        "0.05 for Europa. It is several times the surface value, because the "
        "material is denser with depth.\n"
        "YAML key: thermophysical.kd_override"
    ),
    "thermophysical.H_override": (
        "The depth over which the surface material compacts, in meters: over "
        "this distance the density rises from its loose surface value toward "
        "its deep value. Smaller means the change happens nearer the top.\n"
        "Leave the tick set for the engine's value -- 0.06 m for the Moon, "
        "0.07 m for Europa. Change it to model a thicker or thinner blanket "
        "of loose material.\n"
        "YAML key: thermophysical.H_override"
    ),
    "thermophysical.emissivity": (
        "How efficiently the surface radiates heat away compared with a "
        "perfect blackbody, between 0 and 1. It appears in both the surface "
        "energy balance and the temperature the model finally reports.\n"
        "0.9 to 0.95 covers essentially every planetary regolith; the "
        "defaults are 0.95 for the Moon and 0.9 for Europa.\n"
        "YAML key: thermophysical.emissivity"
    ),
    "thermophysical.incidence_angle_albedo": (
        "Which model describes how the albedo rises as the Sun gets lower in "
        "the sky. Grazing sunlight is reflected more efficiently than "
        "overhead sunlight, which cools the terminator and the poles.\n"
        "Hayne (2017): the lunar angular-albedo fit, and the default.\n"
        "Hapke: the directional photometric model, opt-in.\n"
        "This is a physics choice, not a display option -- it changes the "
        "absorbed flux and therefore the temperatures.\n"
        "It only bites on a body that HAS angular-albedo coefficients, which "
        "among the built-ins means the Moon and Mercury. On Europa and the "
        "others the albedo is flat with incidence angle under either setting, "
        "so this selector changes nothing there.\n"
        "YAML key: thermophysical.incidence_angle_albedo "
        "(Hayne = false, Hapke = true)"
    ),
    "thermophysical.ks": (
        "Thermal conductivity of the loose material at the very surface, in "
        "W/m/K. Together with density and heat capacity it sets the thermal "
        "inertia, which controls how far the daily temperature swings.\n"
        "Regolith is a remarkable insulator: 1e-4 to 1e-2 W/m/K, with the "
        "lunar default 7.4e-4.\n"
        "Raise it toward 1 for exposed bedrock or solid ice.\n"
        "YAML key: thermophysical.ks"
    ),
    "thermophysical.kd": (
        "Thermal conductivity well below the surface, in W/m/K, where the "
        "material has been compacted. Normally several times the surface "
        "value.\n"
        "The lunar default is 3.4e-3.\n"
        "The transition between the two is governed by the scale height "
        "below.\n"
        "YAML key: thermophysical.kd"
    ),
    "thermophysical.H": (
        "The depth over which density and conductivity climb from their "
        "surface values toward their deep values, in meters. It is the "
        "e-folding depth of that transition.\n"
        "0.02 to 0.1 m for planetary regolith; the lunar default is 0.06 m.\n"
        "It is the main knob for matching an observed cooling curve after "
        "sunset, because it sets how quickly the surface starts feeling the "
        "denser material underneath.\n"
        "YAML key: thermophysical.H"
    ),
    "thermophysical.chi": (
        "Strength of radiative heat transfer across the gaps between grains, "
        "which grows sharply with temperature: the conductivity used is "
        "k = k_c x [1 + chi x (T/350)^3].\n"
        "2.7 for lunar regolith; 0 for ice, where the effect is negligible.\n"
        "It matters most where the surface is hot -- it is what makes a "
        "sunlit regolith conduct several times better at noon than at "
        "night.\n"
        "YAML key: thermophysical.chi"
    ),
    "thermophysical.rhos": (
        "Bulk density of the material at the very surface, in kg/m^3, "
        "including its pore space.\n"
        "Porous regolith runs 500 to 1500 kg/m^3; the lunar default is "
        "1100.\n"
        "YAML key: thermophysical.rhos"
    ),
    "thermophysical.rhod": (
        "Bulk density well below the surface, in kg/m^3, after compaction. "
        "Higher than the surface value, and reached over the scale height "
        "above.\n"
        "The lunar default is 1800 kg/m^3.\n"
        "YAML key: thermophysical.rhod"
    ),
    "thermophysical.cp0": (
        "The constant term of the heat capacity, in J/kg/K. The full heat "
        "capacity is temperature-dependent: this constant plus the body's own "
        "polynomial in temperature.\n"
        "Leave it at the body default unless you are modeling an unusual "
        "material. It is negative for the Moon because the lunar polynomial "
        "is fitted that way -- the heat capacity itself stays positive over "
        "the temperature range that occurs.\n"
        "YAML key: thermophysical.cp0"
    ),
    "thermophysical.Qb": (
        "Heat arriving from the interior at the base of the column, in W/m^2. "
        "It sets the temperature a never-lit surface finally reaches.\n"
        "About 0.018 W/m^2 for the Moon, 0.03 for Europa.\n"
        "YAML key: thermophysical.Qb"
    ),
    # The ten "Use Default" checkboxes all say the same thing, so they
    # share one entry: see "ui.use_body_default" below. Ten copies of a
    # sentence is ten places to fix when the sentence is wrong.
    "thermophysical.conductivity_model": (
        "Which law relates thermal conductivity to temperature and density.\n"
        "hayne2017 (default): k = k_c x [1 + chi x (T/350)^3], calibrated on "
        "lunar regolith, where conductivity depends on temperature alone at a "
        "given depth.\n"
        "martinez2021: the Martinez et al. (2021) formulation, in which "
        "conductivity depends on density as well as temperature.\n"
        "Keep the default unless you are specifically reproducing work built "
        "on the other law.\n"
        "YAML key: thermophysical.conductivity_model"
    ),
    "thermophysical.thermal_inertia": (
        "Thermal inertia, sqrt(conductivity x density x heat capacity), in SI "
        "units. It is the single number that best predicts how far the "
        "surface temperature swings between day and night: low inertia means "
        "a hot noon and a cold night.\n"
        "Roughly 20 to 50 for lunar regolith, 50 to 200 for a rocky asteroid, "
        "200 to 2500 for bare rock or solid ice.\n"
        "Typing a value here switches the run to CONSTANT thermal inertia, "
        "which is a much bigger change than it looks. To make the inertia "
        "constant in DEPTH as well as in temperature the engine flattens the "
        "whole regolith profile: the deep conductivity and deep density are "
        "collapsed onto the surface values (on the Moon that is kd 3.4e-3 -> "
        "7.4e-4 and rho_d 1800 -> 1100), the heat capacity is replaced by the "
        "constant TI^2 / (ks x rho_s), and the radiative term chi is forced to "
        "0. Any chi set alongside it is dropped without an error.\n"
        "Use it when you want to match an inertia-fitted measurement, and read "
        "the run's 'Constant thermal inertia mode' banner to see the values it "
        "settled on.\n"
        "To get the layered, temperature-dependent physics back, re-pick the "
        "body or material in the Preset list above. Do not simply type 0: the "
        "panel's own handler still zeroes the radiative term and the heat "
        "capacity on the way past.\n"
        "YAML key: thermophysical.constant_thermal_inertia"
    ),

    # =========================================================================
    # RADIATION TAB
    # =========================================================================
    "radiation.enabled": (
        "Let faces exchange heat with each other: infrared emitted by one "
        "face landing on another, and sunlight bouncing between them. This is "
        "what warms a shadowed crater floor and keeps a polar wall from "
        "radiating straight to space.\n"
        "Off by default because it is the expensive part of the model and a "
        "smooth global sphere barely needs it. Turn it on for craters, rough "
        "terrain, and any polar or cold-trap study -- it is the difference "
        "between a shadowed floor at the geothermal floor and one warmed by "
        "its walls.\n"
        "YAML key: radiation.enabled"
    ),
    "radiation.verbose": (
        "Print detailed progress from the radiation stage to the console. "
        "Useful when a view-factor build seems stuck; noisy otherwise.\n"
        "YAML key: radiation.verbose"
    ),
    "radiation.use_gpu": (
        "Run the radiosity solve, and the view-factor work that feeds it, on "
        "the graphics card. A speed choice; the physics is the same.\n"
        "Off by default. Turn it on for large meshes, where it is worth "
        "several times the wall clock of the processor path.\n"
        "YAML key: radiation.use_gpu"
    ),
    "radiation.min_shadow_fraction": (
        "How much of the time a face must be shadowed before it is allowed to "
        "keep the neighbor infrared the radiosity solve computed for it.\n"
        "IGNORED in a normal configuration. The engine reads it only when "
        "radiation.exchange_selection is set to 'shadow', and the default is "
        "'all' -- every face keeps its flux. This panel does not offer that "
        "key, so changing this field has no effect unless you set "
        "exchange_selection by hand in the YAML.\n"
        "The 'shadow' selection exists only to reproduce runs published "
        "before the default changed. It measures the shadow fraction over the "
        "run's own illumination series, so splitting a run into pieces "
        "reclassifies faces and zeroes their neighbor infrared.\n"
        "YAML key: radiation.min_shadow_fraction"
    ),
    "radiation.beaming": (
        "Account for roughness SMALLER than one mesh facet, which makes a "
        "facet send its infrared preferentially back toward the Sun and the "
        "observer rather than radiating as a smooth disk (Smith 1967). It "
        "raises daytime brightness temperatures and is what reconciles "
        "modeled and measured infrared over rough ground.\n"
        "Off by default. Turn it on -- and set the slope below, since "
        "beaming with a zero slope is refused at startup -- for lunar polar "
        "site and regional meshes, whose facets are meters to hundreds of "
        "meters across and genuinely do not resolve centimeter roughness.\n"
        "Leave it OFF on centimeter-scale and finer meshes: there the mesh "
        "already resolves the roughness, and beaming counts the same energy "
        "twice.\n"
        "It also does nothing at all while radiation exchange is off.\n"
        "YAML key: radiation.beaming"
    ),
    "radiation.rms_slope_deg": (
        "Root-mean-square slope of the roughness living below one facet, in "
        "degrees. It is the only knob the beaming model has: bigger means "
        "rougher and more strongly beamed.\n"
        "Published values run 20 to 37 degrees -- 32.5 for lunar highlands "
        "(Jhoti et al. 2024, the value validated for polar work), about 20 "
        "for the global regolith fit (Bandfield et al. 2015), about 37 "
        "(Rubanenko et al. 2020).\n"
        "Inert while beaming is off. Beaming on with a slope of 0 is a "
        "startup error, not a quiet fall back to smooth emission.\n"
        "YAML key: radiation.rms_slope_deg"
    ),
    "radiation.coupling_mode": (
        "How the loop that alternates between solving temperatures and "
        "solving the radiation exchange decides it has converged.\n"
        "bounded (default): stop once the largest temperature change on any "
        "face falls under the tolerance below. That bounds the error on every "
        "face, which is what makes the answer independent of how the run was "
        "split into chunks.\n"
        "fixed: never stop early -- run exactly the iteration count below, "
        "every window. The most reproducible option, and the most expensive.\n"
        "legacy: stop once only a small fraction of faces are still changing. "
        "That has no error bound, and it made a coupled run's answer depend "
        "on how the time window happened to be split into pieces. It is the "
        "only mode that reads Face Fraction; use it to reproduce a published "
        "run.\n"
        "Spelled exactly: the engine refuses 'Bounded' at startup.\n"
        "YAML key: radiation.coupling_mode"
    ),
    "radiation.max_coupling_iterations": (
        "The most times the temperature solve and the radiation solve may "
        "alternate before the loop gives up.\n"
        "16 is the default. In 'bounded' mode it is a ceiling, not a target: "
        "a typical run converges well inside it and stops on the temperature "
        "tolerance below. In 'fixed' mode it is the exact number of sweeps.\n"
        "Do not lower it to save time. On the large lunar mesh the "
        "sweep-to-sweep change first drops below 1 K at sweep 16, so a cap of "
        "10 truncates the loop and a cap of 5 truncates it badly -- and a "
        "truncated coupled run comes out COLD, by as much as 26 K on the "
        "worst face.\n"
        "YAML key: radiation.max_coupling_iterations"
    ),
    "radiation.coupling_temp_threshold": (
        "How small the largest temperature change between two coupling "
        "iterations must get before the loop is called converged, in kelvin. "
        "This is the ONLY data-driven stop criterion the loop has.\n"
        "0.5 K is the default and is a sensible bar for surface work. "
        "Tighten it for deep-column or cold-trap results, where the answer "
        "itself is a few kelvin.\n"
        "YAML key: radiation.coupling_temp_threshold"
    ),
    "radiation.coupling_flux_threshold": (
        "How much neighbor infrared a face must be receiving before it is "
        "picked for another thermal solve, in W/m^2. It selects faces; it is "
        "NOT a stopping rule, and tightening it will not make the coupled "
        "solution more converged.\n"
        "IGNORED in a normal configuration: it is read only under the "
        "non-default radiation.coupling_select_mode: legacy, a key this panel "
        "does not offer. The default face selection is physics-based and does "
        "not consult it.\n"
        "0.1 W/m^2 is the default. To make the coupling loop stricter, use the "
        "temperature tolerance above.\n"
        "YAML key: radiation.coupling_flux_threshold"
    ),
    "radiation.coupling_face_fraction": (
        "The fraction of faces still changing at which the legacy stop "
        "criterion calls the coupling loop done.\n"
        "IGNORED unless the coupling mode above is 'legacy'. That criterion "
        "carries no error bound and made a run's answer depend on how the "
        "time window was split into pieces, which is why it is not the "
        "default.\n"
        "Change it only alongside coupling_mode: legacy. The engine warns if "
        "you set it without that.\n"
        "YAML key: radiation.coupling_face_fraction"
    ),
    "radiation.view_factor_file": (
        "Where to save, or load, the matrix of view factors -- how much of "
        "each face's sky is filled by every other face. Building it is "
        "usually the most expensive step of a coupled run, and it depends "
        "only on the geometry.\n"
        "Point repeat runs on the same mesh at the same file and the build "
        "happens once.\n"
        "MIND THE CHECK: the engine compares the FACE COUNT only, and on a "
        "mismatch it warns and rebuilds rather than stopping. Two different "
        "meshes with the same number of faces -- the same terrain re-placed at "
        "another latitude, a re-meshed core -- will load silently and the run "
        "will use the wrong geometry. Keep one file per mesh.\n"
        "YAML key: radiation.view_factor_file"
    ),
    "radiation.use_hmatrix": (
        "Store the view factors in compressed, hierarchical form instead of "
        "as one dense matrix, by approximating blocks of faces that are far "
        "apart with a low-rank summary. It turns an impossible memory "
        "requirement into a feasible one.\n"
        "Worth enabling from roughly 50,000 faces up. Beyond about a million, "
        "the sparse route below is usually the better answer.\n"
        "YAML key: radiation.use_hmatrix"
    ),
    "radiation.use_aca": (
        "How each compressed block is built: by adaptive cross approximation, "
        "which samples selected rows and columns, rather than by a full "
        "decomposition of the block.\n"
        "On by default, and much faster for the same tolerance.\n"
        "Only has an effect when hierarchical compression is on.\n"
        "YAML key: radiation.use_aca"
    ),
    "radiation.aca_tolerance": (
        "How accurately each compressed block must reproduce the exact view "
        "factors, as a relative error.\n"
        "1e-3 is the default. Smaller is more accurate and uses more memory.\n"
        "Only has an effect when hierarchical compression is on.\n"
        "YAML key: radiation.aca_tolerance"
    ),
    "radiation.aca_max_rank": (
        "The largest rank a compressed block may reach before the engine "
        "gives up on that block and splits it into smaller ones instead. "
        "(It does not fall back to storing the block in full -- that was "
        "removed as a way to run out of memory.)\n"
        "50 is the default. It is a safety ceiling for blocks that turn out "
        "not to be low-rank, not an accuracy knob.\n"
        "YAML key: radiation.aca_max_rank"
    ),
    "radiation.hmatrix_threads": (
        "How many cores build the compressed view-factor matrix.\n"
        "0 (the default) uses all of them. Lower it if the build runs out of "
        "memory: each thread holds working space of its own.\n"
        "YAML key: radiation.hmatrix_threads"
    ),
    "radiation.nearfield_quadrature_order": (
        "How finely the view factor between two nearly touching facets is "
        "integrated. Order 1 places a single sample at each facet's center; "
        "higher orders integrate symmetrically over both facet areas. The "
        "single-point estimate is badly wrong exactly where facets touch, "
        "which is where the exchange is strongest.\n"
        "3 is the default and the validated production value. Against the "
        "exact answer for two touching facets, order 3 reads 18.5 percent "
        "high and order 1 reads 51.7 percent high.\n"
        "Order 1 is a diagnostic setting, not a production one -- use it only "
        "to isolate what the near field contributes. Going above 3 costs "
        "build time for accuracy that has not been shown to matter.\n"
        "YAML key: radiation.nearfield_quadrature_order"
    ),
    "radiation.use_monte_carlo_vf": (
        "Build the view factors by casting rays from each face and counting "
        "what they hit, instead of integrating the geometry analytically.\n"
        "Off by default. The ray-cast builder is the independent referee: use "
        "it to check the analytic result, or on geometry the analytic path "
        "handles poorly.\n"
        "A different builder is a different answer, not just a different "
        "speed -- do not switch it mid-campaign.\n"
        "YAML key: radiation.use_monte_carlo_vf"
    ),
    "radiation.mc_rays_per_face": (
        "How many rays each face casts when the ray-cast builder is "
        "selected. The statistical noise falls as the square root of this, so "
        "four times the rays halves the error.\n"
        "100,000 is the default. Raise it when checking a result you intend "
        "to publish.\n"
        "Used only when the ray-cast builder above is on.\n"
        "YAML key: radiation.mc_rays_per_face"
    ),
    "radiation.adaptive_neighbors": (
        "Choose each face's radiation partners individually, keeping the "
        "nearest ones until they account for the target share of the energy "
        "that face exchanges, rather than applying one fixed cutoff "
        "everywhere.\n"
        "On by default. It is what lets a mesh of more than a million faces "
        "be handled at all, and it spends the storage where the geometry "
        "actually needs it.\n"
        "YAML key: radiation.adaptive_neighbors"
    ),
    "radiation.captured_energy_target": (
        "What share of each face's radiative exchange the selected partners "
        "must account for.\n"
        "0.95 -- that is, 95 percent -- is the default, and the remaining few "
        "percent is spread over very many distant faces.\n"
        "Raise it toward 0.99 for a deep cold-trap study, where the last few "
        "percent of wall infrared is a large part of the answer; the cost "
        "climbs steeply.\n"
        "YAML key: radiation.captured_energy_target"
    ),
    "radiation.max_neighbors": (
        "A hard ceiling on how many partners any one face may keep, whatever "
        "the energy target asks for.\n"
        "500 is the default. It bounds memory on very rough terrain, where a "
        "face can see a great many others.\n"
        "YAML key: radiation.max_neighbors"
    ),
    "radiation.min_neighbors": (
        "A floor on how many partners every face keeps, so a face that looks "
        "isolated is not left exchanging with nothing.\n"
        "20 is the default.\n"
        "YAML key: radiation.min_neighbors"
    ),
    "radiation.gpu_devices": (
        "How many graphics cards the radiosity solve may spread across.\n"
        "1 (the default) uses the first; 0 uses every one. Ignored when an "
        "explicit device list is given.\n"
        "YAML key: radiation.gpu_devices"
    ),
    "radiation.gpu_device_ids": (
        "Exactly which graphics cards the radiosity solve may use, by index, "
        "as a comma-separated list such as 0,1. Overrides the count above.\n"
        "Leave blank unless you need to keep a specific card free.\n"
        "YAML key: radiation.gpu_device_ids"
    ),
    "radiation.gpu_batch_size": (
        "How many rows of the exchange matrix are sent to the card at once.\n"
        "0 (the default) lets the engine choose.\n"
        "A single block of graphics memory cannot exceed 2 GiB, which is why "
        "large matrices have to be sent in pieces at all. Lower it if a large "
        "matrix fails to upload; it does not change the answer.\n"
        "YAML key: radiation.gpu_batch_size"
    ),
    "radiation.gpu_inter_batch_pause_ms": (
        "An idle gap between radiosity batches, in milliseconds.\n"
        "0 (the default) means no gap. A pause of a few tens of milliseconds "
        "is a first-line mitigation when the display driver resets the card "
        "during a long solve. It costs wall clock and nothing else.\n"
        "YAML key: radiation.gpu_inter_batch_pause_ms"
    ),

    # =========================================================================
    # ILLUMINATION TAB
    # =========================================================================
    "illumination.include_primary_reflection": (
        "Add the sunlight the planet reflects onto its moon -- for Europa, "
        "the light of Jupiter.\n"
        "On by default, and it should stay on for any satellite: on the "
        "sub-planet hemisphere it is a real term in the energy budget.\n"
        "Irrelevant for a body that orbits the Sun directly.\n"
        "YAML key: illumination.include_primary_reflection"
    ),
    "illumination.use_batch_processing": (
        "Trace shadow rays in batches rather than one at a time, which keeps "
        "memory bounded on large meshes.\n"
        "On by default. A performance setting only.\n"
        "YAML key: illumination.use_batch_processing"
    ),
    "illumination.albedo_map": (
        "A map giving Bond albedo face by face, so brightness can vary across "
        "the surface instead of taking one value everywhere. It overrides the "
        "uniform albedo on the Properties tab.\n"
        "Used in Custom mode. Choose Built-in above to have the shipped map "
        "for this body resolved for you.\n"
        "YAML key: illumination.albedo_map"
    ),
    "illumination.albedo_maps_directory": (
        "A folder of albedo maps, from which the one matching the mesh "
        "resolution is chosen. Used in Custom mode.\n"
        "YAML key: illumination.albedo_maps_directory"
    ),
    "illumination.positions_file": (
        "Pre-computed Sun and planet positions for each output time, used "
        "instead of querying the ephemeris during the run.\n"
        "Leave blank for normal use. It exists for reproducing a run whose "
        "geometry must match exactly, or for running without ephemeris "
        "kernels available.\n"
        "YAML key: illumination.positions_file"
    ),
    "illumination.bvh_cache_file": (
        "Where to keep the spatial index that makes shadow ray tracing fast. "
        "Building it depends only on the mesh, so caching it saves real time "
        "on repeat runs of the same geometry, and it is rebuilt "
        "automatically if the mesh no longer matches.\n"
        "READ ONLY WHEN 'Use GPU for Illumination' is on, which is off by "
        "default and recommended off on this machine -- so in the recommended "
        "configuration this field does nothing.\n"
        "Blank (the default) puts it under the output directory. The staleness "
        "check compares the triangle COUNT, so a different mesh with the same "
        "number of triangles would be accepted.\n"
        "The exact lowercase word 'off' disables caching. The engine compares "
        "that string literally, so 'OFF' is treated as a FILENAME, not as the "
        "switch.\n"
        "YAML key: illumination.bvh_cache_file"
    ),
    "illumination.use_gpu": (
        "Trace shadow rays on the graphics card instead of the processor.\n"
        "OFF is the recommendation here, and it is a measured one rather than "
        "an oversight: on this machine the processor path is both correct AND "
        "faster, and the graphics path has a history of lunar correctness "
        "problems. This control exists so the choice is visible rather than "
        "mysterious.\n"
        "Turn it on only to investigate that path.\n"
        "YAML key: illumination.use_gpu"
    ),
    "illumination.gpu_devices": (
        "How many graphics cards the illumination pass may use.\n"
        "Read only while the switch above is on.\n"
        "YAML key: illumination.gpu_devices"
    ),
    "illumination.gpu_device_ids": (
        "Exactly which graphics cards the illumination pass may use, by "
        "index, as a comma-separated list such as 0,1. Overrides the count "
        "above.\n"
        "YAML key: illumination.gpu_device_ids"
    ),
    "illumination.gpu_inter_batch_pause_ms": (
        "An idle gap between ray-tracing batches, in milliseconds -- the same "
        "driver-reset mitigation offered on the other GPU stages.\n"
        "0 (the default) means no gap.\n"
        "YAML key: illumination.gpu_inter_batch_pause_ms"
    ),

    # =========================================================================
    # OUTPUT TAB
    # =========================================================================
    "output.directory": (
        "Where the run writes its frames, statistics and logs. Created if it "
        "does not exist.\n"
        "Give each run its own folder: a second run pointed at the same one "
        "will mix its output with the first.\n"
        "YAML key: output.directory"
    ),
    "output.format": (
        "The form the per-frame fields are written in.\n"
        "binary (default): compact and fast, and what the analysis tooling "
        "reads.\n"
        "vtk: viewer-ready, much larger.\n"
        "none: write no field output at all -- for a run you only want "
        "statistics or location time series from.\n"
        "Spelled exactly in lowercase: the engine silently writes nothing at "
        "all for an unrecognized value.\n"
        "YAML key: output.format"
    ),
    "output.vtk_mode": (
        "Which VTK files to write when the format above is 'vtk'.\n"
        "thermal: temperature only. complete: every field. both: separate "
        "files for each. none: no VTK.\n"
        "'thermal' keeps the output far smaller; choose 'complete' when you "
        "need fluxes and illumination alongside the temperatures.\n"
        "YAML key: output.vtk_mode"
    ),
    "output.include_primary_ir": (
        "Include the infrared arriving from the planet, as its own field, in "
        "the VTK output.\n"
        "Off by default. Turn it on when you need to separate that term from "
        "the rest of the energy budget in a satellite run.\n"
        "YAML key: output.include_primary_ir"
    ),
    "output.locations_file": (
        "A list of latitude and longitude points at which to write full "
        "temperature time series, at every depth -- far more detail than the "
        "per-frame maps carry, at a handful of places.\n"
        "Set it when you have specific sites to compare against "
        "measurements.\n"
        "YAML key: output.locations_file"
    ),
    "output.depths_file": (
        "The depths, in meters, at which those location time series are "
        "reported. Values between the model's own layers are interpolated.\n"
        "Set it alongside the locations file; without it you get the model's "
        "native layer depths.\n"
        "YAML key: output.depths_file"
    ),
    "output.series_output": (
        "Filename for the per-timestep illumination and temperature series at "
        "the points listed in the locations file above.\n"
        "Blank does NOT switch it off -- it means the default name, "
        "illumination_series.csv, inside the output directory. The series is "
        "written whenever a locations file is set, and not otherwise.\n"
        "Change it only to keep several runs' series side by side.\n"
        "YAML key: output.series_output"
    ),

    # =========================================================================
    # GPU SAFETY TAB
    # =========================================================================
    "gpu.margin_explicit": (
        "Write the VRAM floor below into the configuration file.\n"
        "The engine treats the PRESENCE of that key as the request, not its "
        "value: written, the floor is enforced on every platform; absent, it "
        "is enforced on this machine's combination of macOS and AMD hardware "
        "and is advisory elsewhere. Two identical numbers, two different "
        "behaviors, and nothing in the file shows which one you have -- which "
        "is why this is a separate tick.\n"
        "Leave it off unless you mean it. That is what keeps a configuration "
        "this panel has touched from silently changing GPU policy on someone "
        "else's machine.\n"
        "YAML key: gpu.min_vram_margin_gib (written, or not)"
    ),
    #: Shown in place of the entry above while the floor is non-default. It
    #: carries one `{engine_default}` placeholder, filled in by the panel.
    "gpu.margin_explicit.forced": (
        "Forced ON: the floor differs from the engine's own default "
        "({engine_default} GiB), so the key must be written -- and the engine "
        "reads its PRESENCE as 'enforce on every platform'. There is no way "
        "to set a non-default floor advisorily.\n"
        "Put the floor back to the default to make this a free choice again."
    ),
    "gpu.min_vram_margin_gib": (
        "How much graphics memory must stay free on a card, in GiB, after a "
        "resident copy of the radiation exchange matrix has been placed on "
        "it. A card with less headroom has its work separated or refused.\n"
        "2.0 GiB is the default and the validated value on this hardware, "
        "where running a card to the edge of its memory can hang it hard "
        "enough to require a reboot.\n"
        "0 restores the older behavior of checking only that the data fits.\n"
        "YAML key: gpu.min_vram_margin_gib"
    ),
    "gpu.placement_policy": (
        "Whether the resident radiation-exchange copy and the thermal solver "
        "may share the same graphics cards.\n"
        "auto (default): share them, but separate the two onto different "
        "cards once free memory falls below the placement margin below.\n"
        "shared: always share.\n"
        "disjoint: always keep them on separate cards.\n"
        "Keep the default. The engine refuses any other spelling at "
        "startup.\n"
        "YAML key: gpu.placement_policy"
    ),
    "gpu.placement_margin_gib": (
        "The free-memory level, in GiB, below which 'auto' stops letting the "
        "two workloads share a card.\n"
        "4.0 GiB is the default.\n"
        "YAML key: gpu.placement_margin_gib"
    ),
    "gpu.thin_margin_policy": (
        "What actually happens when free memory falls below the floor.\n"
        "separate (default): keep using the graphics cards, but with the two "
        "workloads held apart, refusing only below the hard floor. This is "
        "the recommended setting -- measured on a large lunar site, the "
        "processor fallback cost 108.9 minutes per window against 31.4 on the "
        "cards, while separating them costs only 7 to 10 minutes.\n"
        "refuse: give up on the cards entirely and fall back to the "
        "processor.\n"
        "Choose 'refuse' only if you would rather a run be slow than risk the "
        "card at all.\n"
        "YAML key: gpu.thin_margin_policy"
    ),
    "gpu.hard_min_vram_margin_gib": (
        "The free-memory level, in GiB, below which even 'separate' gives up "
        "and refuses to use the card.\n"
        "0.25 GiB is the default.\n"
        "It must stay BELOW the VRAM floor above, or the band in which "
        "'separate' applies is empty and the engine refuses the whole "
        "configuration at startup.\n"
        "YAML key: gpu.hard_min_vram_margin_gib"
    ),

    # =========================================================================
    # BODY DEFINITION TAB
    # =========================================================================
    "generic_body.ephemeris_source": (
        "How the body's position in space is computed at each time step.\n"
        "auto (default): use SPICE kernels when a metakernel is available, "
        "otherwise propagate the orbital elements below.\n"
        "keplerian: always propagate the orbital elements. Reproducible "
        "anywhere, and accurate enough for a thermal study of a small body.\n"
        "spice: always use SPICE, which needs a metakernel and a NAIF body "
        "identifier. The most accurate option, and the one to use when the "
        "epoch must match a spacecraft observation.\n"
        "Note that 'auto' falls back to Keplerian for any epoch outside the "
        "kernel's coverage.\n"
        "YAML key: generic_body.ephemeris_source"
    ),
    "generic_body.spice_metakernel": (
        "The SPICE metakernel (.tm) listing the kernels for this body. "
        "Required when the ephemeris source is 'spice'.\n"
        "The path typed here may be as long as you like. The 80-character "
        "limit people run into is on the PATH_VALUES lines INSIDE the "
        "metakernel, which have to be split with a '+' continuation; see the "
        "configuration guide if a kernel fails to load from a deep "
        "directory.\n"
        "YAML key: generic_body.spice_metakernel"
    ),
    "generic_body.naif_id": (
        "The identifier SPICE uses to find this body in the loaded kernels. "
        "Required when the ephemeris source is 'spice'.\n"
        "READ IT OUT OF THE KERNEL rather than working it out from the "
        "asteroid number. The published mapping has exceptions, and the "
        "kernel is the only authority on what it actually contains -- the "
        "shipped Bennu example carries 2101955 with that note attached. If "
        "the identifier is wrong the run does not silently fall back; SPICE "
        "reports that it cannot find the body.\n"
        "YAML key: generic_body.naif_id"
    ),
    "generic_body.shape.type": (
        "How the body's shape is represented.\n"
        "spherical: a single radius. Right for a first look, or for a body "
        "with no shape model.\n"
        "ellipsoidal: three semi-axes, which captures overall elongation.\n"
        "custom_mesh: a shape model you provide, which is what you need for "
        "real self-shadowing on an irregular body.\n"
        "YAML key: generic_body.shape.type"
    ),
    "generic_body.shape.radius_km": (
        "Mean radius of the body in kilometers. It sets the physical size of "
        "a spherical body and the reference size used when scaling a shape "
        "model.\n"
        "YAML key: generic_body.shape.radius_km"
    ),
    "generic_body.shape.semi_axis_a_km": (
        "The longest semi-axis of the ellipsoid, in kilometers, along the "
        "body-fixed X direction.\n"
        "YAML key: generic_body.shape.semi_axes_km"
    ),
    "generic_body.shape.semi_axis_b_km": (
        "The intermediate semi-axis of the ellipsoid, in kilometers, along "
        "the body-fixed Y direction.\n"
        "YAML key: generic_body.shape.semi_axes_km"
    ),
    "generic_body.shape.semi_axis_c_km": (
        "The shortest semi-axis of the ellipsoid, in kilometers, along the "
        "spin axis.\n"
        "YAML key: generic_body.shape.semi_axes_km"
    ),
    "generic_body.shape.mesh_file": (
        "The shape model to use, in OBJ or PLY form -- for example a model "
        "from the DAMIT database.\n"
        "Check its units before running: asteroid shape models are usually in "
        "kilometers, and the scale below has to make that right.\n"
        "YAML key: generic_body.shape.mesh_file"
    ),
    "generic_body.shape.mesh_scale": (
        "Multiplier applied to the shape model's coordinates, used to bring "
        "it to the size you want.\n"
        "1.0 when the file is already correctly scaled.\n"
        "YAML key: generic_body.shape.mesh_scale"
    ),
    "generic_body.orbit.semi_major_axis_au": (
        "The orbit's semi-major axis in astronomical units -- its mean "
        "distance from the Sun.\n"
        "This is the single strongest control on temperature, because "
        "sunlight falls off as the inverse square of distance. It also fixes "
        "the orbital period through Kepler's third law.\n"
        "YAML key: generic_body.orbit.semi_major_axis_au"
    ),
    "generic_body.orbit.eccentricity": (
        "How far the orbit departs from a circle: 0 is circular, values "
        "approaching 1 are highly elongated.\n"
        "It sets how much the sunlight varies around the orbit -- the flux "
        "ratio between perihelion and aphelion is ((1+e)/(1-e))^2, so even a "
        "modest eccentricity matters for a seasonal study.\n"
        "YAML key: generic_body.orbit.eccentricity"
    ),
    "generic_body.orbit.inclination_deg": (
        "The tilt of the orbit plane away from the ecliptic, in degrees. With "
        "the ascending node it fixes the orbit normal, which the obliquity is "
        "measured against.\n"
        "YAML key: generic_body.orbit.inclination_deg"
    ),
    "generic_body.orbit.lon_ascending_node_deg": (
        "Where the orbit crosses the ecliptic going north, measured in "
        "degrees around the ecliptic from the reference direction. It orients "
        "the tilted orbit in space.\n"
        "YAML key: generic_body.orbit.lon_ascending_node_deg"
    ),
    "generic_body.orbit.arg_perihelion_deg": (
        "The angle, in degrees, from the ascending node to the point of "
        "closest approach to the Sun. Together with the obliquity it decides "
        "which hemisphere gets its summer at perihelion.\n"
        "YAML key: generic_body.orbit.arg_perihelion_deg"
    ),
    "generic_body.orbit.mean_anomaly_deg": (
        "Where the body sits in its orbit at the reference epoch, in degrees, "
        "as an evenly-advancing angle: 0 at perihelion, 180 at aphelion.\n"
        "This and the epoch below are what tie the orbit to the calendar.\n"
        "YAML key: generic_body.orbit.mean_anomaly_deg"
    ),
    "generic_body.orbit.epoch": (
        "The date and time the orbital elements above describe, in ISO 8601 "
        "form -- for example 2000-01-01T12:00:00 for J2000.\n"
        "Take it from the same source as the elements. Elements applied at "
        "the wrong epoch place the body somewhere else in its orbit.\n"
        "YAML key: generic_body.orbit.epoch"
    ),
    "generic_body.spin.period_hours": (
        "How long the body takes to turn once with respect to the stars, in "
        "hours. It sets the length of the day, and with it the depth the "
        "daily temperature wave reaches: the skin depth grows as the square "
        "root of the period.\n"
        "A fast rotator has a shallow, mild daily cycle; a slow one bakes and "
        "freezes deeply.\n"
        "YAML key: generic_body.spin.period_hours"
    ),
    "generic_body.spin.obliquity": (
        "The angle between the spin axis and the orbit normal, from 0 to 180 "
        "degrees. 0 means the axis is upright to the orbit and there are no "
        "seasons; values near 90 give extreme ones, with each pole facing the "
        "Sun directly in turn.\n"
        "Editing this moves the pole direction to match, keeping the "
        "direction of tilt as it was."
    ),
    "generic_body.spin.pole_ra_deg": (
        "Right ascension of the north spin pole in degrees, in the "
        "Earth-equatorial J2000 frame -- the convention shape-model and "
        "lightcurve papers publish poles in.\n"
        "With the declination it fixes the spin axis, and therefore the "
        "obliquity and the seasons.\n"
        "YAML key: generic_body.spin.pole_ra_deg"
    ),
    "generic_body.spin.pole_dec_deg": (
        "Declination of the north spin pole in degrees, in the "
        "Earth-equatorial J2000 frame.\n"
        "YAML key: generic_body.spin.pole_dec_deg"
    ),
    "generic_body.spin.prime_meridian_deg": (
        "The rotation angle of the body about its own axis at the reference "
        "epoch, in degrees. It sets which longitude faces the Sun when the "
        "run begins.\n"
        "It matters only when a particular longitude has to line up with an "
        "observation; for a statistical study any value will do.\n"
        "YAML key: generic_body.spin.prime_meridian_deg"
    ),

    # =========================================================================
    # ADVANCED
    # =========================================================================
    "advanced.verbose": (
        "Print detailed progress and diagnostics to the console. Useful when "
        "a run behaves unexpectedly; noisy otherwise.\n"
        "YAML key: advanced.verbose"
    ),
    "advanced.default_visible_albedo": (
        "NOT READ BY THE CURRENT ENGINE. The value is parsed out of the "
        "configuration and then never consulted: the visible albedo the run "
        "uses is taken from the body's own properties, or from the albedo map "
        "if one is loaded.\n"
        "Leave it at -1. It is shown rather than hidden so that a "
        "configuration carrying a value does not look as though it is doing "
        "something.\n"
        "YAML key: advanced.default_visible_albedo"
    ),

    # =========================================================================
    # CONTROLS WITH NO CONFIGURATION KEY OF THEIR OWN
    # =========================================================================
    "ui.thermophysical_preset": (
        "Fill the whole Properties tab with a published parameter set for a "
        "known body or material type -- a starting point, not a constraint.\n"
        "Choose the one closest to your target, then adjust individual "
        "values. It overwrites every property below it."
    ),
    "ui.topo_resolved_path": (
        "The topography dataset that Built-in mode will actually load for the "
        "current body and mesh resolution. Read-only.\n"
        "If it is missing from disk, the Fetch button beside it downloads it."
    ),
    "ui.albedo_resolved_path": (
        "The albedo map that Built-in mode will actually load for the current "
        "body and mesh resolution. Read-only.\n"
        "If it is missing from disk, the Fetch button beside it downloads it."
    ),
    "ui.beaming_scale_acknowledgment": (
        "Beaming models roughness the mesh does NOT resolve. This mesh's "
        "facets were measured to be at or below the roughness scale, which "
        "means the mesh already resolves those slopes and beaming would count "
        "the same energy twice.\n"
        "Tick this only if you know the facet-size measurement is misleading "
        "for your mesh -- for example a mesh whose fine facets are confined "
        "to a small patch."
    ),
    "ui.equilibration_units": (
        "Units used to display the spin-up period. Changing it converts the "
        "number beside it and does not change the run."
    ),
    "ui.use_body_default": (
        "Take this property from the selected body's own published values "
        "rather than from the number beside it.\n"
        "Untick to enter your own. The body defaults are the values the "
        "engine would use if this configuration said nothing at all."
    ),
    #: Carries one `{overridable}` placeholder, filled in by the panel with the
    #: properties that DO have an engine path on a built-in body.
    "ui.property_unsupported_on_builtin_body": (
        "Not adjustable for a built-in body. The engine has override keys for "
        "{overridable} and for nothing else, so a value typed here would be "
        "read from the configuration and then silently ignored.\n"
        "To set it, define the surface as a Custom or HORIZONS body, where "
        "the whole property block is honored.\n"
        "Worth knowing either way: only the Moon, Europa and Mercury have "
        "properties of their own in the engine. A built-in name it does not "
        "recognize falls back to Europa's values and says so once in the log."
    ),
    "ui.topo_source_builtin": (
        "Use the topography that ships with PTM3D for this body and mesh "
        "resolution, resolved automatically from the data directory. "
        "Recommended: nothing to locate, and it is the dataset the shipped "
        "examples were built against. It is downloaded on demand if missing."
    ),
    "ui.topo_source_custom": (
        "Use your own folder of pre-built topography grids instead of the "
        "shipped dataset. Choose this when you have prepared elevation data "
        "the built-in set does not cover."
    ),
    "ui.albedo_source_builtin": (
        "Use the Bond-albedo map that ships with PTM3D for this body and mesh "
        "resolution, resolved automatically from the data directory. "
        "Recommended for lunar and Europan work: real albedo contrast changes "
        "absorbed sunlight and therefore temperature. Downloaded on demand."
    ),
    "ui.albedo_source_custom": (
        "Use your own albedo map, as a single file or a folder of maps by "
        "resolution. Choose this when you have a map the shipped set does not "
        "cover."
    ),
    "ui.albedo_source_none": (
        "Use no map at all: every face takes the single Bond albedo from the "
        "Properties tab.\n"
        "Right for an idealized or comparative run where albedo contrast is "
        "not the subject; it removes a real source of temperature variation "
        "on a body that has one."
    ),
    "ui.eq_cadence_mode": (
        "How finely time is sampled during spin-up. Spin-up does not need the "
        "same cadence as the recorded output, and on a large mesh with fine "
        "output this is where most of the illumination cost goes -- decoupling "
        "them can save a large multiple of the run time.\n"
        "Balanced (120 samples per cycle) is the recommended tier. High "
        "accuracy and Fast trade wall clock against roughly 3 K and 7 K of "
        "worst-case difference from the reference. Legacy matches the output "
        "cadence exactly, for reproducing an older run. Custom sets the rate "
        "yourself.\n"
        "It changes spin-up only. Output cadence is always the Time Steps "
        "field on the Simulation tab."
    ),
    "ui.eq_cadence_custom": (
        "Spin-up sampling rate in steps per Earth day, used only in Custom "
        "mode. Prefer one of the named tiers unless you are matching a "
        "specific historical run: the tiers are stated with the accuracy they "
        "were measured to deliver."
    ),
    "ui.eq_cadence_readout": (
        "The spin-up cadence the current settings actually produce for this "
        "body. 'Samples per diurnal cycle' is comparable across bodies; "
        "'steps per Earth day' is what the engine uses internally and depends "
        "on how long this body's solar day is."
    ),
    "ui.duration_units": (
        "Units used to display the simulation duration. Changing it converts "
        "the number beside it and does not change the run."
    ),
    "ui.window_units": (
        "Units the run window is given in. Changing it converts the number "
        "beside it and does not change the run.\n"
        "A diurnal period is this body's solar day. An orbital period is one "
        "trip around the Sun, which is also the seasonal cycle: the subsolar "
        "latitude completes one cycle per orbit, so there is no separate "
        "seasonal length to choose. On a moon that is the PLANET's orbit, not "
        "the moon's orbit around its planet.\n"
        "For this body: {}"
    ),
    "ui.step_size_units": (
        "Units used to display the output step size. Changing it converts the "
        "number beside it and does not change the run."
    ),
    "ui.ti_cp_readout": (
        "The heat capacity the thermal inertia beside it was worked out from, "
        "evaluated at the equilibrium temperature below. Read-only.\n"
        "Until you type a thermal inertia this is the body's own "
        "temperature-dependent heat capacity sampled at that temperature. "
        "Once you do, it becomes the CONSTANT value that inertia implies, "
        "TI^2 / (ks x rho_s), and the readout says so."
    ),
    "ui.ti_teq_readout": (
        "The body's global equilibrium temperature for this albedo, "
        "emissivity and distance from the Sun -- the temperature a uniform "
        "sphere reaches when it re-radiates over its whole surface what it "
        "absorbs over its cross-section. Read-only.\n"
        "It is a sanity reference, not a prediction of any one face: a "
        "diurnal run should straddle it, with noon well above and night well "
        "below."
    ),
    "ui.advanced_options": (
        "Reveal the expert groups on every tab: view-factor and coupling "
        "options, the GPU device controls and guardrails, mesh placement and "
        "occluder, initial condition and run state, the equilibration "
        "cadence, and the engine-flag editor.\n"
        "Every tab is visible either way; this changes what is shown inside "
        "them, never what is saved."
    ),
    "ui.yaml_preview": (
        "The fully expanded configuration handed to the engine, and written "
        "beside the output directory as config.yaml.\n"
        "Save does NOT write this: it edits your project file in place, "
        "changing only the keys you changed and preserving its comments."
    ),
    "ui.step_size": (
        "The time between recorded frames -- the same information as Time "
        "Steps, expressed the other way round. Editing either updates the "
        "other.\n"
        "Aim for at least about 48 frames per diurnal cycle: the surface "
        "temperature peak is sharp, and a coarser cadence clips it."
    ),

    # =========================================================================
    # THE BODY-FIRST SCREENS
    # =========================================================================
    "ui.body_button": (
        "Choose the body you are modeling. The choice sets the orbit, the "
        "spin, the surface properties, the ephemeris kernels and which "
        "physics options are even offered, so the rest of the setup is a "
        "handful of decisions rather than a form.\n"
        "Moon has the fullest data support: an albedo map, a topography "
        "catalog, the sub-facet roughness calibration and every worked "
        "example. Europa adds the five Galileo stereo elevation models and "
        "the forty-five shipped site configurations. Mercury is runnable but "
        "ships no map, no topography grid and no worked example. Asteroid and "
        "Custom body both write a generic_body: section: use Asteroid when the "
        "body is defined from an orbit, a spin pole and a shape model, and "
        "Custom body when it is defined from its surface properties.\n"
        "Only bodies this engine actually implements appear here. A body it "
        "has no properties for is refused rather than run with another "
        "body's radius and no sunlight, so use Custom body for anything not "
        "shown."
    ),
    "ui.body_mode": (
        "Whether to model the whole body as a sphere, or one site from a "
        "terrain mesh you supply.\n"
        "Global is right for diurnal and seasonal behavior and for global "
        "maps. Local mesh is right for craters, ridges and landing sites, "
        "where local shadowing and infrared exchange between facets "
        "dominate.\n"
        "This is the engine's own distinction and it is the mesh file alone: "
        "supplying mesh.import_file makes it a local run."
    ),
    "ui.run_type": (
        "What the run should produce.\n"
        "Time series writes one field per output sample, so any face can be "
        "plotted against time. Recommended for a first run and for anything "
        "diurnal.\n"
        "Integrated statistics writes the minimum, mean and maximum over the "
        "window and skips every per-sample file -- the same physics, a "
        "fraction of the disk. Right for long or seasonal runs.\n"
        "Parameter grid repeats the same setup once per point of a grid.\n"
        "Sets output.skip_time_series."
    ),
    "ui.all_settings": (
        "Open the full eleven-section configuration view: Simulation, Mesh, "
        "Properties, Thermal, Output, Illumination, Radiation, GPU and the "
        "rest. Everything a card shows is there too, along with every option a "
        "card leaves at its default.\n"
        "You do not need it for an ordinary run -- the body cards cover that, "
        "and this is here for when you already know which setting you want. "
        "It stays out of the way until you ask for it.\n"
        "Nothing is lost by moving between the two: they edit the same "
        "configuration, an edit made in either is visible in the other, and "
        "the Back button returns you to the card you came from."
    ),
    "ui.close_all_settings": (
        "Close the full settings view and go back to where you were in the "
        "body-first flow -- the same body, the same run type, the same card.\n"
        "Nothing is discarded. Anything you changed here is already in the "
        "configuration, and the card re-reads it on the way back so the two "
        "always show the same values."
    ),
    "ui.back_to_bodies": (
        "Return to the body chooser. The configuration is not reset -- "
        "everything set so far stays set until another body's card is "
        "applied, which announces what it changed."
    ),
    "ui.back_to_runtype": (
        "Return to the run-type step, where the choice between the global "
        "domain and a local mesh also lives. The configuration is left as it "
        "is."
    ),
    "ui.landing_open": (
        "Open an existing configuration file. It must be YAML and it must "
        "name a body under simulation.body, because that is what decides "
        "which card describes it.\n"
        "Anything the card does not show is kept exactly as the file had it, "
        "listed above the card, and editable under All settings."
    ),
    "ui.landing_baseline": (
        "Load the configuration a finished run actually used, from that run's "
        "own config_used.yaml, and report the engine build it ran on.\n"
        "Use this to reproduce a published result: it reads the values rather "
        "than guessing them, and it says plainly whether your engine is the "
        "same one. Point it at the run's output directory.\n"
        "Beaming: unchanged. Everything the baseline recorded is loaded as it "
        "stands -- its spin-up seed, its coupling mode and its beaming setting "
        "included -- and nothing here overrides any of it.\n"
        "The engine environment is the one thing it may not be able to tell "
        "you. A run's record does not list the PTM_* variables it used, so if "
        "the loaded configuration carries no engine_env: section -- a run "
        "started from the command line usually does not -- that part is "
        "unknown and your current settings are used instead. It says which of "
        "the two cases you are in after it loads."
    ),
    "ui.europa_terrain": (
        "Which Europa digital elevation model to run. The list, the file "
        "paths and the latitudes each terrain offers are read out of the "
        "configurations shipped with PTM3D, so they cannot drift from them.\n"
        "The five terrains are SOCET SET stereo elevation models built from "
        "Galileo Solid State Imager images at the U.S. Geological Survey "
        "Astrogeology Science Center, published under CC BY 4.0 as Bland, "
        "Michael (2022), \"Europa DEMs and Orthoimages in Support of Bland et "
        "al. 2021 (Remote Sensing)\", figshare, "
        "doi:10.6084/m9.figshare.16904632, supporting Bland et al. (2021), "
        "Remote Sensing 13, 5097. The deposit has since superseded them with "
        "versions registered to Europa's cartographic coordinate system, and "
        "asks that these be used to check that paper's conclusions -- so they "
        "carry the relief, not a current mapped placement. The fuller note is "
        "in data/external/README.md.\n"
        "Choosing one fills in both the terrain mesh and the matching "
        "view-factor cache path. These elevation models are not distributed "
        "with PTM3D; the line below says whether the file is on disk.\n"
        "Sets mesh.import_file and radiation.view_factor_file."
    ),
    "ui.split_windows": (
        "Run the span as several SHORT windows at different epochs instead of "
        "one continuous integration. Each window is a complete run with its "
        "own spin-up, so the set samples the long span at a fraction of the "
        "cost -- and no temperature state is carried from one window to the "
        "next, so they are samples of the span and not an integration of it.\n"
        "Recommended for anything longer than about a hundred diurnal cycles. "
        "That is the pattern the lunar A3GT campaign uses across the 18.6-year "
        "nodal cycle, and it keeps each run inside the single-dispatch "
        "envelope this machine tolerates.\n"
        "Leave it off for a run of a few diurnal or orbital periods, where one "
        "continuous window is both cheaper and more informative.\n"
        "Writes no configuration key: a window series is N separate runs, each "
        "with its own simulation.start_time and simulation.duration_days."
    ),
    "ui.window_count": (
        "How many windows the span is sampled with. Each one is a separate "
        "run and costs what one run of that length costs, so this multiplies "
        "the wall clock directly.\n"
        "Four is a reasonable start and is the size of the A3GT campaign's own "
        "production year set. More windows resolve the seasonal cycle better; "
        "fewer finish sooner.\n"
        "The windows are placed at EQUAL intervals across the span. The A3GT "
        "campaign instead puts its windows on the annual subsolar-latitude "
        "extremes, found with SPICE by scripts/a3gt_v2_subsolar.py; that "
        "script and the campaign orchestrator remain the tools for that."
    ),
    "ui.window_span": (
        "The total stretch of time the windows are spread across -- not the "
        "length of a run. The first window starts at the run's start time and "
        "the last one ends at the end of this span.\n"
        "For a seasonal study make it one orbital period or more, since that "
        "is the period of the subsolar latitude. The 18.6 years the lunar "
        "A3GT campaign spans is the Moon's nodal cycle, over which the phase "
        "of the annual subsolar-latitude maximum drifts all the way around.\n"
        "It must be at least the number of windows times the window length, "
        "or the windows would overlap and count part of the span twice."
    ),
    "ui.window_series": (
        "Write one configuration per window, plus a windows.json recording "
        "the schedule, how the epochs were chosen, and the command to run "
        "each window.\n"
        "They are written, not launched. The commands use "
        "scripts/run_chunked_simulation.py -- the same runner the A3GT "
        "campaign orchestrator invokes for each of its windows -- and a "
        "multi-hour set belongs in a detached process that survives closing "
        "this window."
    ),
    "ui.europa_sweep": (
        "Write one configuration per point of the terrain-by-latitude grid -- "
        "the same set of runs the shipped Europa configurations describe, "
        "derived from the choices on this card.\n"
        "They are written, not launched. A grid of this size runs for hours, "
        "and a long run belongs in a detached process that survives closing "
        "this window."
    ),
    "ui.ephemeris_keplerian": (
        "Compute the body's position from orbital elements rather than from a "
        "SPICE kernel. A deliberate choice, recorded in the configuration.\n"
        "Right when no kernel covers your dates, or when you are varying the "
        "orbit yourself. Accurate for a short window near the elements' own "
        "epoch and progressively less so away from it."
    ),
    "ui.ephemeris_spice": (
        "Compute the body's position from a SPICE kernel. The accurate "
        "choice, when the kernel covers every epoch the run needs.\n"
        "Check the coverage before trusting it: for an epoch the kernel does "
        "not cover, the engine falls back to orbital elements for that epoch "
        "and says nothing, so a run can be entirely element-based while its "
        "record says SPICE."
    ),
    "ui.spk_coverage": (
        "Read the chosen kernel's own coverage window and compare it with "
        "this run.\n"
        "Worth doing every time: the fallback to orbital elements happens per "
        "epoch, silently, and the only way to know it did not happen is to "
        "check that the kernel covers the whole window -- including the "
        "spin-up, which runs before the start date."
    ),
    "ui.body_lookup": (
        "Look the name in the Name field up and fill in what is known about "
        "that body: NAIF id, radius, spin period, pole and orbital "
        "elements.\n"
        "Recommended before typing any of those by hand -- a mistyped "
        "semi-major axis is a value nothing downstream can range-check.\n"
        "The local cache under data/horizons_cache is searched first and "
        "needs no network; JPL HORIZONS is asked only if the cache has no "
        "match. If neither answers, nothing is changed and the message says "
        "which happened. Everything filled in stays editable, and if the body "
        "brings a SPICE kernel its coverage is compared with this run's dates "
        "straight away."
    ),
    "ui.fetch_spk": (
        "Ask JPL HORIZONS to build a SPICE kernel for this body over this "
        "run's dates, and store it under data/horizons_cache so later runs "
        "need no network.\n"
        "The request reaches a year past each end of the run, including the "
        "spin-up that happens before the start date, because a kernel costs "
        "the same whether it spans a month or a decade and a short one fails "
        "invisibly.\n"
        "What arrives is read back and compared with the run before anything "
        "is changed. If it does not span the whole window, the ephemeris "
        "source is left as it was and the reason is printed: for an epoch "
        "outside a kernel the engine substitutes orbital elements and logs "
        "nothing, so a kernel that was never checked is a run that was never "
        "checked.\n"
        "Nothing is downloaded until you press this."
    ),
    "ui.fetch_sbdb": (
        "Fill in radius, Bond albedo, rotation period and spin pole from "
        "JPL's Small-Body Database.\n"
        "Two of those are converted rather than copied, because the database "
        "and the model do not use the same definitions. The database "
        "publishes a diameter and a GEOMETRIC albedo p_V; this model takes a "
        "radius and a BOND albedo, and it publishes the spin pole in ecliptic "
        "coordinates while this model takes right ascension and declination. "
        "The line under each field shows the conversion.\n"
        "Thermal inertia is not filled in: no database publishes it the way "
        "this one publishes a diameter, so the conductivity, density and heat "
        "capacity below stay yours to set.\n"
        "Everything filled in stays editable, and a failure changes nothing."
    ),
    "ui.fetch_damit_shape": (
        "Look this body up in DAMIT, the database of asteroid shape models "
        "derived from lightcurves, and use its model if one exists.\n"
        "Most asteroids have no model, and the answer says so rather than "
        "substituting a near miss: a different asteroid number is a different "
        "body.\n"
        "A model brings its own spin pole and rotation period, and they are "
        "offered with it. They belong together -- the shape is described in "
        "the frame its own pole and period define, so keeping the shape while "
        "using a pole from elsewhere points the body in the wrong direction. "
        "If the two disagree, the card says so instead of choosing.\n"
        "The model fixes the body's shape but not its size, so it is scaled "
        "to the radius on this card as it is stored."
    ),
    "ui.shape_provenance": (
        "Where the shape file came from, and how to get it again.\n"
        "Downloaded shape models are stored beside a small text record of "
        "their source, database id and checksum. The models themselves are "
        "excluded from this project's version control by a rule written for "
        "compiler output, so a model exists in one copy on this machine and "
        "the record is what makes it recoverable."
    ),
    "ui.spin_provenance": (
        "Where each spin value came from: a database, a shape model, or you.\n"
        "Worth reading before a long run. A pole from a lightcurve inversion "
        "is a fit with an uncertainty of a few degrees and sometimes a mirror "
        "solution; a pole nobody has measured is whatever was left in the "
        "field. Neither is marked in the output, and both run to completion."
    ),
    "ui.thermal_inertia_manual": (
        "Thermal inertia is not filled in from any database, and this says "
        "what published values look like so the fields below can be set by "
        "hand.\n"
        "The values that exist come from individual thermophysical fits to "
        "individual infrared data sets, each with its own roughness and shape "
        "assumptions, so one lifted into a different model is not a "
        "measurement. Fine regolith on a large main-belt asteroid runs low, "
        "tens of units; a kilometer-scale near-Earth rubble pile runs in the "
        "hundreds; bare rock runs over a thousand."
    ),
    "ui.orbit_on_body_page": (
        "Open the Body page, where the six orbital elements and their epoch "
        "live.\n"
        "They are not repeated here on purpose: one value edited in two "
        "places is a value whose stored form depends on which page was "
        "displayed last."
    ),

    # =========================================================================
    # KEYS THE CARDS EXPOSE THAT THE SECTION TABS REACH ANOTHER WAY
    # =========================================================================
    "illumination.albedo_source": (
        "Where the per-face Bond albedo comes from.\n"
        "Built-in map uses the albedo map shipped for this body, which is "
        "recommended for lunar and Europan work because real albedo contrast "
        "changes absorbed sunlight and therefore temperature.\n"
        "Custom uses a map of your own.\n"
        "None gives every face the single albedo from the Properties page, "
        "which is right for an idealized or comparative run.\n"
        "A built-in per-face map is matched to the procedural sphere it was "
        "built for by face count, so it cannot apply to an imported terrain "
        "mesh; the line beneath says what will actually be loaded."
    ),
    "thermal.use_default_property_map": (
        "Give each face its own H parameter from the map bundled for this "
        "body and sphere resolution, instead of one value everywhere.\n"
        "Recommended on the Moon, where the bundled map carries the measured "
        "regolith variation. It applies only to the procedural sphere it was "
        "built for; an imported terrain mesh uses the single value from the "
        "Properties page.\n"
        "Sets thermal.use_default_property_map."
    ),
    "output.skip_time_series": (
        "Skip the per-sample field files and keep only the summary maps "
        "(minimum, mean and maximum over the window).\n"
        "The physics is identical either way; this decides only what is "
        "written. Turn it on for long or seasonal runs, where the per-sample "
        "files are most of the disk used and the summary is what gets "
        "analyzed.\n"
        "Sets output.skip_time_series."
    ),
    "generic_body.name": (
        "What to call this body. It names the run in logs and provenance and "
        "is what a fetched HORIZONS record fills in.\n"
        "It is not looked up anywhere: the physics comes from the orbit, spin, "
        "shape and surface values below, not from the name.\n"
        "Sets generic_body.name."
    ),

    # =========================================================================
    # A CUSTOM BODY'S SURFACE
    # -------------------------------------------------------------------------
    # These are a DIFFERENT set of keys from the Properties page's, and they
    # behave differently: on a built-in body most of the plainly-named
    # properties are read by nothing, while on a custom body this whole block
    # is honored. They are described here rather than pointed at the other
    # entries so that neither description has to hedge about the other case.
    #
    # Every one of them carries the same floor. A value below 0.0000005 is
    # written to the solver with six decimal places, arrives as 0.000000, fails
    # the engine's positive-value check, and is silently replaced by a
    # placeholder property -- so the card refuses those rather than letting the
    # run use a number nobody chose.
    # =========================================================================
    "generic_body.surface.albedo": (
        "The fraction of sunlight this surface reflects across all "
        "wavelengths (the Bond albedo). It sets how much sunlight is absorbed, "
        "so it drives the temperature more directly than any other property "
        "here.\n"
        "Typical values: about 0.04 for a dark carbonaceous asteroid, 0.2 for "
        "a stony one, 0.5 or more for fresh ice.\n"
        "NOT the geometric albedo p_V that catalogs publish. The two differ by "
        "the phase integral q, and the conversion is A = p_V x q with "
        "q = 0.290 + 0.684 G, where G is the slope parameter of the H, G "
        "magnitude system (0.15 when none is published). Typing a published "
        "p_V straight in reflects far too much sunlight and runs the body "
        "cold: for a bright asteroid with p_V 0.42 and G 0.32 the Bond albedo "
        "is 0.21, and using 0.42 instead absorbs a quarter less sunlight. "
        "Filling this in from the Small-Body Database applies the conversion "
        "and shows it under the field.\n"
        "Sets generic_body.surface.albedo."
    ),
    "generic_body.surface.emissivity": (
        "How efficiently the surface radiates heat away in the infrared, as a "
        "fraction of a perfect blackbody.\n"
        "Recommended 0.9 to 0.95 for any natural regolith; the value is "
        "poorly constrained for most small bodies and the results are not very "
        "sensitive to it.\n"
        "Sets generic_body.surface.emissivity."
    ),
    "generic_body.surface.ks": (
        "Thermal conductivity of the topmost, most porous layer, in watts per "
        "meter per kelvin. With density and heat capacity it sets the thermal "
        "inertia, which is what controls how far the daily heat wave "
        "penetrates and how much the surface cools at night.\n"
        "Typical regolith values are 0.001 to 0.01; solid rock is near 2.\n"
        "Must be at least 0.0000005: below that the value does not survive "
        "the trip into the solver and a placeholder is used instead.\n"
        "Sets generic_body.surface.ks."
    ),
    "generic_body.surface.kd": (
        "Thermal conductivity of the deeper, compacted material, in watts per "
        "meter per kelvin. The model blends from the surface value to this one "
        "over the depth scale H.\n"
        "Usually several times larger than the surface value, because "
        "compaction closes the pore space that insulates the top layer.\n"
        "Must be at least 0.0000005, for the same reason as the surface "
        "value.\n"
        "Sets generic_body.surface.kd."
    ),
    "generic_body.surface.rhos": (
        "Bulk density of the topmost layer, in kilograms per cubic meter. "
        "Part of the thermal inertia, and of how much heat the near-surface "
        "can store.\n"
        "Typical loose regolith is 1000 to 1500; ice-rich material is lower.\n"
        "Sets generic_body.surface.rhos."
    ),
    "generic_body.surface.rhod": (
        "Bulk density of the deeper material, in kilograms per cubic meter, "
        "blended from the surface value over the depth scale H.\n"
        "Typically 1.5 to 2 times the surface value on a regolith-covered "
        "body.\n"
        "Sets generic_body.surface.rhod."
    ),
    "generic_body.surface.cp0": (
        "Specific heat capacity, in joules per kilogram per kelvin -- how much "
        "energy it takes to warm the material by one degree. The third factor "
        "in the thermal inertia.\n"
        "Around 800 for silicate regolith at room temperature; it falls "
        "steeply at the low temperatures of the outer solar system, which is "
        "what the temperature-dependent form on the Properties page is for.\n"
        "Sets generic_body.surface.cp0."
    ),
    "generic_body.surface.H": (
        "The depth over which the surface properties blend into the deep ones, "
        "in meters. Small H means the transition happens right at the top and "
        "the insulating layer is thin.\n"
        "Around 0.06 m on the Moon; there is no measured value for most small "
        "bodies, so it is usually left at a lunar-like scale.\n"
        "Sets generic_body.surface.H."
    ),
    "generic_body.surface.Qb": (
        "Heat flowing up from the interior, in watts per square meter. It sets "
        "the floor the deepest layer settles to, which matters only where the "
        "sun never reaches.\n"
        "About 0.018 on the Moon. On a small body it is effectively zero, so "
        "leave it at zero unless you have a value for this body.\n"
        "Sets generic_body.surface.Qb."
    ),
    "generic_body.surface.chi": (
        "How strongly radiation through the pore space adds to conduction as "
        "the material warms -- the coefficient on the cubic temperature term.\n"
        "Around 2.7 for lunar regolith. Set it to zero to model conduction "
        "alone, which is what a constant thermal inertia means.\n"
        "Sets generic_body.surface.chi."
    ),

    # =========================================================================
    # FLUX CUBES  (docs/flux-cubes.md)
    # =========================================================================
    "flux_cubes.enabled": (
        "Stands small virtual cubes on the terrain and records the sunlight "
        "and infrared arriving on each of their six faces -- up, down, north, "
        "south, east and west -- at every output step. It is what a lander leg "
        "or a radiator would see, split into direct sunlight, sunlight "
        "scattered off the surrounding ground, infrared from that ground, and "
        "the two the primary body contributes.\n"
        "The cubes only watch. They cast no shadow, emit nothing, and are not "
        "part of the surface the model solves, so every temperature and every "
        "flux on the mesh comes out the same whether this is on or off. The "
        "cost is about one percent of the run.\n"
        "Needs facet-to-facet infrared exchange on, and an imported terrain "
        "mesh. Sets flux_cubes.enabled."
    ),
    "flux_cubes.mode": (
        "How the cubes are laid out on the ground.\n"
        "grid: a regular pattern of cubes covering a circle around the center "
        "of the mesh. Use it to map how the local sky and ground vary across a "
        "site.\n"
        "named locations: a cube at each latitude and longitude you list, and "
        "no grid. Use it for specific points -- a candidate landing pad, an "
        "instrument site -- where the results are keyed by the name you gave.\n"
        "grid and named locations: both in one layout, the named points added "
        "after the grid.\n"
        "The cubes are placed by the layout tool, not by this file. This "
        "control decides what the tool is asked to build, and which points the "
        "run checks the layout for."
    ),
    "flux_cubes.spacing_m": (
        "How far apart neighboring cubes stand, in meters, on the grid.\n"
        "25 m is the value the published site products used. The cube count "
        "grows as the inverse square, so halving this quadruples both the "
        "layout time and the size of the result.\n"
        "Used when the layout is built. The layout file records it, and the "
        "run reads it back from there."
    ),
    "flux_cubes.radius_m": (
        "How far the grid reaches from the center of the mesh, in meters. "
        "Every grid point inside this distance gets a cube.\n"
        "500 m is the value the published site products used, giving 1,257 "
        "cubes at 25 m spacing. Keep it inside the detailed part of the mesh: "
        "cubes that reach the coarse surrounding terrain see a sky that is not "
        "resolved there.\n"
        "Used when the layout is built. The layout file records it, and the "
        "run reads it back from there."
    ),
    "flux_cubes.cache_file": (
        "The file holding where every cube stands and what each of its faces "
        "can see. It is built once by the layout tool -- use the button below "
        "-- and read by the run.\n"
        "It belongs to one mesh. The run compares the layout against its own "
        "mesh and stops if they differ, rather than reporting results for "
        "somewhere else, so rebuild this whenever the terrain mesh changes.\n"
        "Sets flux_cubes.cache_file."
    ),
    "flux_cubes.write_binary": (
        "Writes flux_cubes/flux_cubes.fcb: every cube, face, component and "
        "time step in one file, at full precision.\n"
        "Leave it on. It is the complete result, and the conversion script "
        "that produces the shareable HDF5 file reads it. The spreadsheet "
        "files below are a convenience on top of it.\n"
        "Sets flux_cubes.write_binary."
    ),
    "flux_cubes.write_csv": (
        "Also writes one spreadsheet file per cube, under flux_cubes/csv, with "
        "a row per output step and a column per face and component.\n"
        "Convenient for a handful of cubes and unwieldy for a full grid, which "
        "is what the limit below is for. Named locations are keyed by their "
        "name rather than by a number.\n"
        "Sets flux_cubes.write_csv."
    ),
    "flux_cubes.write_csv_max_cubes": (
        "Caps how many grid cubes get their own spreadsheet file. Zero means "
        "all of them, which on a full grid is over a thousand files per run.\n"
        "The cap never drops a named location: those were asked for by name, "
        "so they are always written, and a layout containing any of them "
        "writes every file regardless of this number.\n"
        "Sets flux_cubes.write_csv_max_cubes."
    ),
    "flux_cubes.shadow_probe_rays": (
        "How the run decides whether the sun reaches a cube. Two values, "
        "because the engine reads nothing between them -- anything under five "
        "is treated as one.\n"
        "1: one ray from the center of the cube. Cheapest, and it reports a "
        "cube straddling a shadow edge as fully lit or fully dark.\n"
        "5: the center plus the four horizontal face centers. When those "
        "disagree the step is marked as partly shadowed, which is the flag to "
        "look for near a shadow edge. This is the recommended setting and the "
        "one the published products used.\n"
        "Sets flux_cubes.shadow_probe_rays."
    ),
    "ui.flux_cubes.height": (
        "How high the cube center stands above the ground, and how wide the "
        "cube is. Both are set to a tenth of the local facet size, so the "
        "downward face reads the ground immediately beneath it rather than an "
        "average over a wide patch.\n"
        "This is not a setting: the layout tool measures the mesh and records "
        "what it used. The number here is this window's estimate of what the "
        "tool will find, taken over the whole mesh rather than over the grid "
        "area, so expect it to move by a few percent. Once a layout file "
        "exists, the value shown is the one the tool measured."
    ),
    "ui.flux_cubes.locations": (
        "The named points to put a cube at: a name, a latitude and a longitude "
        "for each, and optionally a height and cube size in meters to override "
        "the automatic ones.\n"
        "Names become the file names in the result, so keep them short and "
        "distinct -- at most 31 characters, and no repeats. Latitude and "
        "longitude are in the body-fixed frame, the same convention the mesh "
        "placement fields above use.\n"
        "Written into the configuration as flux_cubes.locations, where the run "
        "checks that the layout file really contains every name."
    ),
    "ui.flux_cubes.import_csv": (
        "Loads a list of named points from a comma-separated file with the "
        "columns name, latitude, longitude, and optionally height and cube "
        "size in meters. Blank lines and lines beginning with # are skipped, "
        "and a header row is recognized.\n"
        "It replaces whatever is in the table. Any line that cannot be read is "
        "reported rather than skipped, because a point dropped on import is a "
        "cube you would believe was in the result and is not."
    ),
    "ui.flux_cubes.build": (
        "Runs the layout tool on the terrain mesh above and writes the layout "
        "file, using the placement settings shown here. It measures the mesh, "
        "places the cubes, and casts rays from each cube face to record what "
        "that face can see.\n"
        "It uses the processor only -- no graphics card -- and takes tens of "
        "seconds on a site-scale mesh. Placing cubes is this tool's job alone, "
        "which is why editing the settings here does not move a cube until the "
        "layout is rebuilt."
    ),
    "ui.flux_cubes.status": (
        "What the current layout file contains: how many cubes, how they are "
        "arranged, how high they stand, and which mesh they were built for. "
        "Read back from the file itself rather than from the settings above, "
        "so it describes the layout the run will actually use."
    ),
}


# Default thermophysical properties by body (matches planetary_properties.h)
BODY_DEFAULTS = {
    "Europa": {
        "albedo": 0.5,
        "emissivity": 0.9,
        "Qb": 0.03,
        "ks": 0.0096,
        "kd": 0.05,
        "rhos": 500.0,
        "rhod": 700.0,
        "H": 0.07,
        "cp0": 0.0,
        "incidence_angle_albedo": False,
        "chi": 0.0,           # Ice — no radiative conductivity
        "rAU": 5.2,
        "cpCoeff": [7.49],    # cp = 7.49*T (Giauque & Stout 1936)
    },
    "Moon": {
        "albedo": 0.12,
        "emissivity": 0.95,
        "Qb": 0.018,
        "ks": 7.4e-4,
        "kd": 3.4e-3,
        "rhos": 1100.0,
        "rhod": 1800.0,
        "H": 0.06,
        "cp0": -3.6125,
        "incidence_angle_albedo": True,
        "chi": 2.7,           # Hayne et al. 2017
        "rAU": 1.0,
        "cpCoeff": [2.7431, 2.3616e-3, -1.2340e-5, 8.9093e-9],
    },
    "Mercury": {
        "albedo": 0.07,
        "emissivity": 0.93,
        "Qb": 0.02,
        "ks": 0.002,
        "kd": 2.0,
        "rhos": 1800.0,
        "rhod": 2200.0,
        "H": 0.1,
        "cp0": 800.0,
        "incidence_angle_albedo": True,
        "chi": 2.7,
        "rAU": 0.387,
        "cpCoeff": [0.25, 0.0],
    },
    # Bodies not yet fully implemented in C++ - use reasonable defaults
    "Enceladus": {
        "albedo": 0.81,
        "emissivity": 0.9,
        "Qb": 0.025,
        "ks": 0.001,
        "kd": 0.01,
        "rhos": 500.0,
        "rhod": 700.0,
        "H": 0.05,
        "cp0": 0.0,
        "incidence_angle_albedo": False,
        "chi": 0.0,           # Ice
        "rAU": 9.54,
        "cpCoeff": [7.49],
    },
    "Titan": {
        "albedo": 0.22,
        "emissivity": 0.9,
        "Qb": 0.005,
        "ks": 0.02,
        "kd": 0.2,
        "rhos": 500.0,
        "rhod": 700.0,
        "H": 0.1,
        "cp0": 0.0,
        "incidence_angle_albedo": False,
        "chi": 0.0,           # Ice
        "rAU": 9.54,
        "cpCoeff": [7.49],
    },
    "Mars": {
        "albedo": 0.25,
        "emissivity": 0.95,
        "Qb": 0.02,
        "ks": 0.02,
        "kd": 2.0,
        "rhos": 1500.0,
        "rhod": 2000.0,
        "H": 0.1,
        "cp0": 800.0,
        "incidence_angle_albedo": False,
        "chi": 2.7,
        "rAU": 1.524,
        "cpCoeff": [],        # Constant cp = cp0
    },
    "Ceres": {
        "albedo": 0.09,
        "emissivity": 0.95,
        "Qb": 0.01,
        "ks": 0.001,
        "kd": 0.01,
        "rhos": 1200.0,
        "rhod": 1500.0,
        "H": 0.05,
        "cp0": 800.0,
        "incidence_angle_albedo": False,
        "chi": 0.0,           # Ice/clay mixture
        "rAU": 2.77,
        "cpCoeff": [],        # Constant cp = cp0
    },
}


def get_tooltip(key):
    """Get tooltip text for a configuration key."""
    return TOOLTIPS.get(key, "")


def get_label(key, fallback=""):
    """The plain-language display label for a key, or the caller's own.

    Only keys this module deliberately relabels appear in ``LABELS``; every
    other control keeps the label its call site passes. Renaming the
    CONFIGURATION key is never in scope here -- the engine reads those.
    """
    return LABELS.get(key) or fallback


def get_body_defaults(body_name):
    """Get default thermophysical properties for a planetary body."""
    return BODY_DEFAULTS.get(body_name, BODY_DEFAULTS["Europa"])


# Synodic day (solar day) in Earth days for built-in bodies.
#
# For the three bodies the engine IMPLEMENTS these match the C++ thermal model's
# `props.day` exactly, and src/python/tests/test_body_capabilities.py fails if
# they stop matching. That test exists because this table's previous comment
# made the same claim and it was FALSE for Mercury: this table said 175.942 d
# while the engine carried the SIDEREAL 58.646 d and printed it as the solar
# day. Both are now the derived 175.938 d = 3 x the sidereal period
# (BUGS.md ENG-MERCURY-SOLAR-DAY).
#
# The last four are NOT engine values and cannot be: the engine implements no
# properties for those bodies and REFUSES them at config parse
# (ENG-BODY-UNBACKED-RUN). They are kept only so the duration widget can still
# convert units for a config someone typed by hand; they describe the real
# solar system, not this simulator. Phase B of the body-first work removes them
# from the dropdown, at which point they should leave this table too.
BODY_SYNODIC_PERIODS = {
    # --- implemented by the engine; pinned to props.day by the parity test ---
    "Europa": 3.55,         # Synchronous rotation
    "Moon": 29.530588,      # Synodic lunar day
    "Mercury": 175.938,     # Solar day = 3 x the 58.646 d sidereal rotation
    # --- NOT implemented: the engine refuses these body names ---
    "Enceladus": 1.370218,  # Synchronous rotation
    "Titan": 15.945,        # Synchronous rotation
    "Mars": 1.02749,        # Martian sol
    "Ceres": 0.3781,        # ~9.074 hours
}

# Constants for Kepler's third law, matching src/cpp/orbital/orbital_elements.h
# TO THE DIGIT: gm::SUN and OrbitalElements::AU_TO_M.
#
# The AU used to be the rounded 1.496e11 here while the engine used
# 1.495978707e11. Two spellings of one constant, 14 parts per million apart:
# a generic body's DIURNAL period came out of this file and, once the run
# window gained an "orbital periods" unit, its ORBITAL period came out of
# ptm.body_capabilities -- which reads the engine's number. The two would have
# disagreed in the same combo, on the same body, for no reason a user could
# see. The engine's value is the one a run is actually integrated with.
_GM_SUN = 1.32712440018e20  # m^3/s^2
_AU_METERS = 1.495978707e11  # meters per AU


def _as_number(value, fallback):
    """Coerce a YAML scalar for comparison, never raising.

    These helpers read straight from the config, where yaml.safe_load may have
    left a str (a quoted number, or `5e3`), None (a bare `key:`), or a
    non-finite float. A bare `<=` on any of those raises, and every caller here
    is reached from a Qt slot -- where an escaping exception is a process abort
    (PyQt6 qFatal, exit 134), not an error message.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number or number in (float("inf"), float("-inf")):
        return fallback
    return number


def compute_synodic_day_days(config_manager):
    """Compute synodic day from generic_body config data (Kepler + rotation).

    Mirrors the C++ logic in generic_body.h::updateDerivedProperties().
    Returns the synodic day in Earth days, or None if data is unavailable.
    """
    import math

    a_au = _as_number(
        config_manager.get_value("generic_body.orbit.semi_major_axis_au", -1.0),
        -1.0)
    period_hours = _as_number(
        config_manager.get_value("generic_body.spin.period_hours", -1.0), -1.0)

    if a_au <= 0 or period_hours <= 0:
        return None

    a_m = a_au * _AU_METERS
    orbital_period_s = 2.0 * math.pi * math.sqrt(a_m ** 3 / _GM_SUN)
    sidereal_day_s = period_hours * 3600.0

    if abs(sidereal_day_s - orbital_period_s) > 1.0:
        synodic_s = 1.0 / abs(1.0 / sidereal_day_s - 1.0 / orbital_period_s)
    else:
        synodic_s = sidereal_day_s  # Synchronous

    return synodic_s / 86400.0


def get_synodic_period_days(config_manager, body_name=None):
    """Return the synodic day (solar day) in Earth days for the given body.

    Priority: compute from HORIZONS/generic_body config → fallback dict → current config.
    """
    if body_name is None:
        body_name = config_manager.get_value("simulation.body", "Europa")

    body_name = str(body_name or "")
    clean_name = body_name[4:] if body_name.startswith("[H] ") else body_name

    computed = compute_synodic_day_days(config_manager)
    if computed is not None:
        return computed

    if clean_name in BODY_SYNODIC_PERIODS:
        return BODY_SYNODIC_PERIODS[clean_name]

    # The last resort returns the RAW config scalar, which then becomes
    # _current_synodic_days on two tabs and is multiplied and compared there.
    # Coerced for the same reason as everything else in this module: an
    # uncached body (Custom, an uncached [H] ...) plus a quoted or
    # exponent-spelled duration was a process abort.
    return _as_number(
        config_manager.get_value("simulation.duration_days", 3.55), 3.55)


# HELIOCENTRIC orbital period in Earth days -- one trip around the SUN, which
# is also the seasonal cycle (the subsolar latitude completes one cycle per
# orbit). For a moon that is its PLANET's orbit.
#
# It used to say "Matches the C++ thermal model (thermo.year)", and for the
# three moons it did: the engine sets a synchronously rotating moon's `year`
# equal to its DAY (planetary_properties.h:189, "Same as day for synchronous
# rotation"), so Europa read 3.55 d. That is Europa's orbit around Jupiter.
# Meanwhile the body cards' run-window control offers a unit spelled exactly
# "orbital periods" and resolves it from ptm.body_capabilities, which is
# heliocentric -- so one unit name meant 3.55 d on the Thermal page and
# 4332.589 d on the card, a factor of 1220 apart in the same application.
#
# The three implemented bodies are now pinned to
# ptm.body_capabilities.HELIOCENTRIC_PERIOD_DAYS by
# test_body_capabilities.py, which is the only thing that keeps them from
# drifting again. The other four are not engine bodies and are here only so
# the unit toggle still converts for a config someone typed by hand; they
# follow the same definition so the column means one thing.
BODY_ORBITAL_PERIODS = {
    # --- implemented; pinned to HELIOCENTRIC_PERIOD_DAYS by the parity test --
    "Europa": 4332.589,     # Jupiter's orbit around the Sun
    "Moon": 365.25,         # Earth year -- the Moon orbits the Sun with it
    "Mercury": 87.969,      # Mercury year
    # --- NOT implemented: the engine refuses these body names ---
    "Enceladus": 10759.22,  # Saturn's orbit around the Sun
    "Titan": 10759.22,      # Saturn's orbit around the Sun
    "Mars": 686.97,         # Martian year
    "Ceres": 1681.63,       # ~4.6 Earth years
}


def compute_orbital_period_days(config_manager):
    """Compute orbital period from generic_body config via Kepler's third law.

    Returns the orbital period in Earth days, or None if data is unavailable.
    """
    import math

    a_au = _as_number(
        config_manager.get_value("generic_body.orbit.semi_major_axis_au", -1.0),
        -1.0)
    if a_au <= 0:
        return None

    a_m = a_au * _AU_METERS
    orbital_period_s = 2.0 * math.pi * math.sqrt(a_m ** 3 / _GM_SUN)
    return orbital_period_s / 86400.0


def get_orbital_period_days(config_manager, body_name=None):
    """Return the orbital period (year) in Earth days for the given body.

    Priority: compute from HORIZONS/generic_body config → fallback dict → synodic period.
    """
    if body_name is None:
        body_name = config_manager.get_value("simulation.body", "Europa")

    body_name = str(body_name or "")
    clean_name = body_name[4:] if body_name.startswith("[H] ") else body_name

    computed = compute_orbital_period_days(config_manager)
    if computed is not None:
        return computed

    if clean_name in BODY_ORBITAL_PERIODS:
        return BODY_ORBITAL_PERIODS[clean_name]

    # Fallback: use synodic period as a rough estimate
    return get_synodic_period_days(config_manager, body_name)


# ── Obliquity utilities ──────────────────────────────────────────────
# Earth's obliquity for equatorial↔ecliptic conversion (J2000).
# Matches SpinState::OBLIQUITY_J2000_RAD in src/cpp/orbital/spin_state.h.
_OBLIQUITY_J2000_DEG = 23.4392911


def compute_obliquity_deg(pole_ra_deg, pole_dec_deg,
                          inclination_deg, lon_ascending_node_deg):
    """Compute obliquity (angle between spin pole and orbit normal) in degrees.

    Matches C++ formulas in body_fixed_frame.cpp (spin pole equatorial→ecliptic)
    and kepler_propagator.cpp (orbit normal from orbital elements).

    Returns obliquity in [0, 180] degrees, or None if inputs are invalid.
    """
    import math

    # Orbit normal in ecliptic J2000  (kepler_propagator.cpp:28-54)
    i_rad = math.radians(inclination_deg)
    Omega_rad = math.radians(lon_ascending_node_deg)
    nx = math.sin(Omega_rad) * math.sin(i_rad)
    ny = -math.cos(Omega_rad) * math.sin(i_rad)
    nz = math.cos(i_rad)

    # Spin pole: equatorial J2000 → ecliptic J2000  (body_fixed_frame.cpp:15-37)
    ra_rad = math.radians(pole_ra_deg)
    dec_rad = math.radians(pole_dec_deg)
    sx_eq = math.cos(dec_rad) * math.cos(ra_rad)
    sy_eq = math.cos(dec_rad) * math.sin(ra_rad)
    sz_eq = math.sin(dec_rad)

    eps = math.radians(_OBLIQUITY_J2000_DEG)
    ce, se = math.cos(eps), math.sin(eps)
    sx = sx_eq
    sy = sy_eq * ce + sz_eq * se
    sz = -sy_eq * se + sz_eq * ce

    # Obliquity = arccos(n · s)
    dot = nx * sx + ny * sy + nz * sz
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def compute_pole_for_obliquity(obliquity_deg, current_pole_ra_deg, current_pole_dec_deg,
                                inclination_deg, lon_ascending_node_deg):
    """Compute new pole RA/Dec to achieve the given obliquity, preserving tilt direction.

    When changing obliquity, the spin pole moves along the great circle defined by
    the orbit normal and the current tilt direction. If the current spin pole is
    aligned with the orbit normal (obliquity ≈ 0), the ascending node direction
    is used as the default tilt direction.

    Returns (pole_ra_deg, pole_dec_deg).
    """
    import math

    obl_rad = math.radians(obliquity_deg)

    # Orbit normal in ecliptic J2000
    i_rad = math.radians(inclination_deg)
    Omega_rad = math.radians(lon_ascending_node_deg)
    n = [math.sin(Omega_rad) * math.sin(i_rad),
         -math.cos(Omega_rad) * math.sin(i_rad),
         math.cos(i_rad)]

    # Current spin pole in ecliptic
    ra_rad = math.radians(current_pole_ra_deg)
    dec_rad = math.radians(current_pole_dec_deg)
    sx_eq = math.cos(dec_rad) * math.cos(ra_rad)
    sy_eq = math.cos(dec_rad) * math.sin(ra_rad)
    sz_eq = math.sin(dec_rad)

    eps = math.radians(_OBLIQUITY_J2000_DEG)
    ce, se = math.cos(eps), math.sin(eps)
    s = [sx_eq,
         sy_eq * ce + sz_eq * se,
         -sy_eq * se + sz_eq * ce]

    # Perpendicular component: tilt direction
    dot_sn = s[0] * n[0] + s[1] * n[1] + s[2] * n[2]
    e = [s[k] - dot_sn * n[k] for k in range(3)]
    e_mag = math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2)

    if e_mag < 1e-10:
        # Current spin pole ≈ orbit normal → use ascending node direction as default.
        # Ascending node dir = (cos Ω, sin Ω, 0) is always perpendicular to n.
        e = [math.cos(Omega_rad), math.sin(Omega_rad), 0.0]
        e_mag = math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2)

    # Normalize tilt direction
    e = [e[k] / e_mag for k in range(3)]

    # New spin pole in ecliptic
    new_s = [math.cos(obl_rad) * n[k] + math.sin(obl_rad) * e[k] for k in range(3)]

    # Ecliptic → equatorial (inverse rotation by ε about x-axis)
    new_sx_eq = new_s[0]
    new_sy_eq = new_s[1] * ce - new_s[2] * se
    new_sz_eq = new_s[1] * se + new_s[2] * ce

    dec = math.degrees(math.asin(max(-1.0, min(1.0, new_sz_eq))))
    ra = math.degrees(math.atan2(new_sy_eq, new_sx_eq)) % 360.0

    return ra, dec


# ── Thermal Inertia utilities ────────────────────────────────────────
_SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann (W/m²/K⁴), SI-2019 exact; canonical: src/cpp/constants/stefan_boltzmann.h


def compute_equilibrium_temperature(albedo, emissivity, rAU):
    """Compute global equilibrium temperature: T_eq = ((1-A)*S/(4*eps*sigma))^0.25.

    Uses S = 1361 / rAU² for solar constant at heliocentric distance rAU.
    """
    rAU = _as_number(rAU, 0.0)
    emissivity = _as_number(emissivity, 0.0)
    albedo = _as_number(albedo, 0.0)
    if rAU <= 0 or emissivity <= 0:
        return 270.0  # Fallback
    S = 1361.0 / (rAU * rAU)
    return ((1.0 - albedo) * S / (4.0 * emissivity * _SIGMA_SB)) ** 0.25


def compute_cp_at_temperature(T, cp0, cpCoeff):
    """Evaluate cp(T) = cp0 + c1*T + c2*T^2 + ... using Horner-like accumulation.

    Matches the C++ thermal model polynomial convention (planetary_properties.h).
    Returns cp, clamped to a minimum of 1.0 J/kg/K.
    """
    cp = cp0
    T_power = T
    for c in cpCoeff:
        cp += c * T_power
        T_power *= T
    return max(cp, 1.0)


def compute_thermal_inertia(ks, rhos, cp):
    """Compute thermal inertia TI = sqrt(ks * rhos * cp)."""
    import math
    return math.sqrt(max(0.0, ks * rhos * cp))
