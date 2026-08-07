# Tutorial 07 — The PTM3D GUI, end to end

**Application:** [`gui/run_gui.py`](../../gui/run_gui.py) (the thermal simulation
app, `gui/ptm3d_gui.py`)
**Cost of the walkthrough run:** one quickstart-class Moon simulation —
1,280 faces, one lunation, CPU only. Measured 3.8 s of engine time on a 48-core
workstation; expect a minute or two on a laptop.
**Verified:** 2026-08-07 against `origin/main` (7fd214f), PyQt6 6.7.1 / Qt 6.7.1.

This tutorial covers the same ground as the command-line tutorials, through
the graphical front end. The GUI drives the engine binary, so build PTM3D
first ([INSTALL.md](../../INSTALL.md)).

The figures below were rendered from the live panels. The five marked
**[SCREENSHOT-PAUL #n]** remain to be captured on a live display; they are
tabulated at the end.

---

## 1. Launching the GUI

### Dependencies

```bash
cd gui/
pip install -r requirements.txt
```

Required: `PyQt6`, `numpy`, `PyYAML`. Optional but strongly recommended:
`pyvista` + `pyvistaqt` (the 3D view) and `matplotlib` (the 2D plots). The
launcher checks these and reports which are missing (`gui/run_gui.py:42-90`):

```
ERROR: Missing required dependencies:
  - PyQt6

Install with:
  pip install PyQt6
```

A missing *optional* package is a warning, not a failure — the app starts with
that feature disabled.

### Entry point

```bash
cd gui/
python run_gui.py
```

Run it **from the `gui/` directory**. Two optional arguments open the app with
something already loaded:

```bash
python run_gui.py --project /path/to/config.yaml     # open a saved project
python run_gui.py --results /path/to/output_dir/     # open a finished run
```

`run_gui.py` is the supported entry point. `gui/run_dashboard.py` is a
separate multi-module launcher that bundles the DEM/mesh tools and receives
less testing.

On the **first launch only**, the app asks which target body you want
(`gui/ptm3d_gui.py:163-191`). You can change it any time on the Simulation
tab; the prompt does not appear again.

> **[SCREENSHOT-PAUL #1]** — the first-launch "Select target body" dialog. To
> make it appear again, quit the GUI and run
> `defaults delete com.ptm3d.PTM3DSimulator bodyPickerShown`, then relaunch.
> Frame the dialog with the body list open.

---

## 2. The window

Three regions:

- **Left** — the configuration panel (a nav rail of pages, plus a live YAML
  preview underneath) and, as a second tab, the **Results Browser**.
- **Right** — the visualization panel: **3D Visualization** and **2D Plots**.
- **Bottom** — a docked **Console Output** pane, visible from launch.

The divider between left and right is draggable; ``Ctrl+` `` toggles the
console, and **View > Reset Layout** puts everything back.

> **[SCREENSHOT-PAUL #2]** — the whole window immediately after launch, before
> any run: config panel on the left with the Simulation page selected, the empty
> 3D view on the right, the console at the bottom showing "PTM3D Console Ready".
> Frame the full window so the toolbar, status bar, and all three regions are
> visible.

Keyboard shortcuts: `F5` run, `Shift+F5` stop, `Ctrl+T`
validate configuration, `Ctrl+S` save project, `Ctrl+L` load a results
directory, `F6` refresh the 3D view, `Home` reset the camera.

---

## 3. Basic mode — the five pages you need

The configuration panel opens in **Basic** mode: five pages, which is everything
a standard thermal run needs. The **Advanced options** checkbox above the nav
rail reveals the rest (section 7).

### Simulation

![Simulation tab](images/gui/01_basic_simulation_tab.png)

- **Start Time** — ISO 8601 UTC. Drives the SPICE ephemeris lookup.
- **Duration** with a unit selector (days / hours / Earth years / **synodic
  periods**). Selecting a body re-sets this to **one synodic day of that body**
  (Moon: 29.530588 d), so a fresh project always starts at one full diurnal
  cycle.
- **End Time (UTC)** — computed from start + duration, and editable: type an
  end time and the duration follows.
- **Time Steps** and **Step Size** are two views of the same quantity and stay
  in sync — edit either one. These set the **output** cadence.
- **Target Body** — Moon, Europa, Mercury, Enceladus, Titan, Mars, Ceres, plus
  `Custom Body...` and `Browse HORIZONS...` for anything else. Changing the
  body updates every body-aware page at once (properties, mesh datasets, albedo
  maps, equilibration units).

The **YAML preview** at the bottom is live: it is the file that will be handed
to the engine, and the quickest way to confirm a control changed the key you
intended.

### Mesh

![Mesh tab](images/gui/02_basic_mesh_tab.png)

- **Resolution** — icosphere subdivision level. Level 3 is 1,280 faces (the
  quickstart), level 5 is 20,480, level 7 is 327,680. Face count grows as
  `20 x 4^level`; so does everything downstream.
- **Use Topographic Mesh** plus a **Topography source** selector: *Built-in*
  resolves a shipped dataset under `data/topo` (through `PTM_DATA_ROOT` or the
  repository `data/` directory) and shows you the exact resolved path in the
  read-only **Resolves to** box; *Custom path* points at your own directory.
  If a built-in dataset is not on disk, the box says `NOT ON DISK` or
  `DATALESS (cloud placeholder)` and a **Fetch...** button appears.
- **Custom Mesh Import** — an `.obj`/`.ply`/`.vtk` of your own, with a scale
  factor. Setting an import file disables the global-mesh controls, because they
  no longer apply.

### Properties

![Properties tab](images/gui/03_basic_properties_tab.png)

Thermophysical properties, seeded from the selected body. Each row has a
**Use Default** checkbox; uncheck it to type your own value.

> For a **built-in** body the engine accepts custom values only for **Bond
> albedo, emissivity, ks and kd** (written as `*_override` keys). It has no
> path for `H`, `rhos`, `rhod`, `cp0`, `Qb`, or `chi` on a built-in body, so
> the GUI **disables** those six rows
> (`gui/gui_panels/config_panel.py:1828-1845` and `:2201-2238`).
> To customize them, use a Custom or HORIZONS body, where they are written to
> `generic_body.surface:` and are read.

**Incidence-Angle Albedo** is a model selector, not an on/off switch:
*Hayne (2017)* (the default for global bodies) or *Hapke*. It maps to the single
engine key `incidence_angle_albedo` (`false` = Hayne, `true` = Hapke). There is
no "off" — `false` already selects the Hayne angular model.

The **Thermal Inertia** row is derived: it shows TI = sqrt(k rho cp) and the
implied `cp(T_eq)`. Typing a value there activates constant-TI mode, which
**forces chi to 0** (no radiative conductivity) and makes `cp` constant.

### Thermal

![Thermal tab](images/gui/04_basic_thermal_tab.png)

- **Run Thermal Model** — off gives you an illumination-only run.
- **CPU Threads** — 0 = auto (all cores).
- **Equilibration** — spin-up before the output window, in synodic periods by
  default (also selectable in orbital periods, days, hours, Earth years).
- **Compute Statistics** — writes the per-face min/mean/max binaries. Leave it on.
- **Use GPU for Thermal Solver** — only meaningful on a GPU-enabled build
  (section 9).

Under **Advanced thermal** (collapsed by default, expanded in the figure):

- **Equilibration Cadence** — decouples the equilibration time step from the
  output cadence. Presets are stated with their measured accuracy against a
  legacy reference run (Balanced 120 samples/cycle ~4 K max; High accuracy 240
  ~3 K; Fast 30 ~7 K; Legacy = match output cadence, byte-identical to old
  PTM3D). On a large mesh with fine output cadence this is the difference
  between minutes and hours.
- **Integration Window** — leave both at `-1` (automatic) unless you are
  deliberately integrating over a sub-window.
- **Thermal Property Maps** — per-face H or thermal-inertia maps
  (None / bundled-for-body / custom binary file).

### Output

![Output tab](images/gui/05_basic_output_tab.png)

- **Output Directory** — relative paths resolve against the **project root**,
  which the GUI derives from the location of the engine binary (the directory
  containing `build/bin/ptm_thermal_sim`), *not* from wherever you launched the
  GUI. The console prints the pinned working directory (`[run] cwd: ...`) at
  launch.
- **Format** (`binary` or `vtk`) and **VTK Mode** (`thermal` / `complete` /
  `both`).
- **Location Exports** — a `lat,lon` file and a depths file give you
  per-location, per-depth temperature series in addition to the per-face
  binaries.

---

## 4. Your first run

This is [tutorial 02](02_lunar_diurnal_icosphere.md)'s lunar diurnal run, done
through the GUI.

1. **Body**: Moon (the default). Duration stays at one synodic day.
2. **Mesh > Resolution**: `3` (1,280 faces). Start small; you can always go up.
3. **Simulation > Time Steps**: `144`.
4. **Thermal**: leave GPU off; **Compute Statistics** on.
5. **Output > Output Directory**: `output/gui_tutorial_moon_quickstart`.
6. Optional: `Ctrl+T` (**Simulation > Validate Configuration**) for a quick
   sanity check on ranges and required fields.
7. Press **Run** (or `F5`).

Saving first is not required: if a project file is open the GUI saves and runs
it; otherwise it writes an engine-facing `config.yaml` (all paths absolutized)
into the output directory and runs that (`gui/ptm3d_gui.py:679-709`). The
console reports which file it ran.

### The pre-flight check

Before the engine starts, the GUI runs a pre-flight
(`gui/gui_core/preflight.py`) and prints the result to the console:

![Console at launch](images/gui/11_console_preflight.png)

The lines report:

- **which binary** will run, by absolute path, **md5, and build time** — a
  quick check that a rebuild took effect;
- **where the output goes**, and how much disk is free there;
- **which engine environment flags** are being applied — by default the
  recommended physics-fix set (`PTM_EQ_PERFACE_FLUX=1`, `PTM_ILLUM_RISE_HABS=0`,
  `PTM_ILLUM_SEAL_HOLES=1`, `PTM_ILLUM_TRUE_NORMALS=1`).

Pre-flight *blocks* the launch on missing/unreadable files and on "dataless"
cloud placeholders (files evicted to the cloud; reading one hangs the engine
indefinitely). It asks for an explicit override on cloud-synced paths and on a
view-factor rebuild. Keep simulation input data on a local disk.

### The view-factor cost estimate

If radiation exchange is on and no view-factor file is set, the engine builds
the view-factor matrix first — which can dominate the run. The GUI warns
before launch, with an estimate:

![View-factor rebuild dialog](images/gui/10_preflight_view_factor_dialog.png)

The estimate is labeled approximate (disk from 8 bytes per nonzero; solve time
from measured throughput; the *build* time is separate and highly variable)
and advisory — it warns but never blocks. The same estimate updates live on
the Radiation page (section 7).

For this walkthrough radiation is off, so no dialog appears.

### Watching it run

The **Run** button greys out, **Stop** becomes available, a progress bar appears
in the status bar, and the indicator on the right turns yellow **Running**. The
console streams the engine's output.

> **[SCREENSHOT-PAUL #3]** — a run in progress. Start the quickstart run above
> and catch it mid-way. Frame the bottom-right of the window so the status bar
> shows the progress bar partly filled and the yellow "Running" badge, with the
> console streaming "Progress N/144" lines above it.

Two notes on the console:

- Lines are colored by **content**, not by which stream they came from (the
  engine writes informational banners to stderr). Red is reserved for genuine
  failure signatures (`ERROR:`, `FATAL:`, tracebacks, crash signatures,
  "simulation failed"), yellow for warnings
  (`gui/gui_core/simulation_runner.py:33-57`).
- A **watchdog** samples the child process's CPU clock every 30 s. If the
  process stops accumulating CPU time it posts a `[watchdog]` warning and the
  status bar reads `STALLED`, distinguishing a long computation from a hang
  (a blocked file read, a wedged GPU).

### Completion

![Console at completion](images/gui/12_console_completed.png)

On exit code 0 you get a green `Simulation completed successfully`, and the GUI
**loads the output directory automatically** and switches to the Results
Browser. On a nonzero exit code you get a red failure line with the code, and no
auto-load.

Exit code 0 alone does not confirm a healthy run. Check the tail of the log:
face count, time steps, `Statistics export completed successfully`, and a flux
summary that is not all zeros. The run above reported 1,280 faces, 144 time
steps, and a maximum total flux of 1,413.9 W/m^2 — about right for the Moon at
1 AU.

---

## 5. Looking at the results

### Results Browser

![Results browser](images/gui/13_results_browser.png)

The tree shows what the run produced — per-timestep series (temperature,
illumination, direct solar, primary reflected, primary IR), the
`Statistics (annual)` aggregate set, `mesh.ply`, `time_steps.txt`, and
`body_positions.csv` — with file counts and sizes. The **Details** pane
summarizes the directory; the **Timesteps** table lists every output step with
its epoch and UTC label, and clicking a row drives the 3D view.

A directory is only accepted if it contains `mesh.ply`; otherwise the panel
tells you it is not a PTM3D output directory.

### 3D and 2D views

The right-hand panel has two tabs. **3D Visualization** renders the mesh colored
by the selected field, with rotate/zoom/pan, face picking (click a face to get
its values), timestep animation, and image export (`Ctrl+E`). **2D Plots** gives
temperature histograms, temperature-vs-latitude scatter, and time series,
including the series for a picked face.

The toolbar's **Timestep** and **Field** selectors populate from what is
present in the loaded directory — statistics fields from `statistics/*.bin`,
per-step fields from the per-field subdirectories — so only fields the run
wrote are selectable (`gui/ptm3d_gui.py:947-997`).

> **[SCREENSHOT-PAUL #4]** — the 3D Visualization tab after loading
> `output/gui_tutorial_moon_quickstart`, with Field = "Annual Max Temperature
> (K)". Rotate so the sub-solar hemisphere is visible. Frame the 3D view plus
> the toolbar row above it, so the Timestep and Field selectors are readable.

> **[SCREENSHOT-PAUL #5]** — the 2D Plots tab on the same results: the
> temperature histogram and the temperature-vs-latitude scatter. Frame the whole
> right-hand panel.

For exporting to other tools, the **Export Data** button on the visualization
panel writes **GeoTIFF**, **HDF5**, both, or **VTU/PVD** (`ExportDataDialog`,
`gui/gui_panels/visualization_panel.py:2125`); the 2D Plots tab has its own
**Export Data (ASCII)** button for the plotted series. For reading the raw
binaries yourself, see
[06_interpreting_outputs.md](06_interpreting_outputs.md).

---

## 6. Saving and reopening a project

`Ctrl+S` writes the configuration as YAML — the same schema the command-line
engine reads, so a GUI project *is* a config file and vice versa. `Ctrl+O`
reopens one, **File > Open Recent** keeps the last ten. A run's auto-saved
`config.yaml` (in the output directory, with absolutized paths) is the exact
configuration that ran, which makes it the right thing to archive next to the
results.

---

## 7. Advanced mode

Tick **Advanced options** above the nav rail. Four more pages appear —
Illumination, Radiation, Engine Flags, HORIZONS — plus **Body** when a custom or
HORIZONS body is active (it is meaningless for a built-in body, so it stays
hidden).

![Advanced nav rail](images/gui/06_advanced_nav_rail.png)

This is purely a visibility filter. Hidden pages still load and save; toggling
it changes nothing in your YAML.

### Illumination

**Include Reflected Sunlight from Primary Body** gates only the
**reflected-sunlight** term from the primary (Jupiter for Europa, Earth for
the Moon) in the total incident flux
(`src/cpp/illumination/illumination_model.cpp:2279`). It does **not** gate the
primary's **thermal IR**, which is computed independently and zeroed only when
the primary is the Sun (`:2263-2271`). Turning this off is not "no primary
body."

The page also carries the **albedo source** selector: *Built-in* resolves
the shipped Bond-albedo map for the body and resolution and shows the resolved
path, *Custom path* takes your own file or directory, *None* means every face
uses the constant albedo from the Properties tab. Under **Advanced
illumination** are the ray-tracing batch controls and precomputed Sun/primary
position files.

### Radiation

![Radiation tab](images/gui/07_advanced_radiation_coupling.png)

**Enable Radiation Exchange** turns on face-to-face radiosity — necessary for
craters, rough terrain, and anything with self-heating; pointless on a smooth
convex sphere. **Use GPU for Radiosity** and **Min Shadow Fraction** sit
alongside it, and the collapsed sections hold the coupling controls, the
view-factor file/H-matrix/ACA options, and the sparse-view-factor settings for
meshes above ~1M faces.

**Coupling controls.** The engine's default coupling mode is `bounded`
(`src/cpp/tools/config_parser.h:315`). In that mode the **Face Fraction**
control has no effect — it applies only in `legacy` mode, hence its label
**"Face Fraction (legacy mode only)"** (`gui/gui_panels/config_panel.py:1664-1671`);
the face-fraction exit criterion has no error bound, so `bounded` and `fixed`
modes ignore it. The GUI has no control for `coupling_mode` itself: to run
`legacy` (or `fixed`), add `radiation.coupling_mode: legacy` to the YAML by
hand, or pass `--rad-coupling-mode` on the command line. If Face Fraction is
changed without setting `coupling_mode: legacy`, the engine prints a console
warning:

  ```
  WARNING [PTM-89]: radiation.coupling_face_fraction=0.02 is IGNORED in
  coupling_mode='bounded' (the face-fraction exit has no error bound).
  Use coupling_mode: legacy to restore it.
  ```

**Max Coupling Iterations**, **Temperature Tolerance**, and **Flux Tolerance**
are live in the default mode.

The blue box at the bottom of the page is the live **view-factor cost
estimate**, recomputed whenever the mesh or the radiation settings change. It
reports estimated nonzeros, disk, and radiosity solve time per window, with an
explicit band (partners-per-face is a heuristic, 5-20 for smooth convex, ~105
default). At high resolution it adds a warning:

![View-factor estimator warning](images/gui/08_vf_cost_estimator_warning.png)

At level 9 (5,242,880 faces) the estimate crosses the single-GPU radiosity
limit of 536,870,912 nonzeros (a 2^31-byte SpMV offset), beyond which
single-GPU radiosity cannot allocate; the chunked/int64 path or multiple GPUs
are required. The estimator warns and lets you proceed — it is advisory, not a
gate.

If radiation is off, the box reads
`Enable radiation exchange to see a view-factor cost estimate.`

### Engine Flags

![Engine flags tab](images/gui/09_advanced_engine_flags.png)

The engine gates several physics fixes behind `PTM_*` environment variables so
that old runs stay byte-identical. The GUI applies the recommended-ON set by
default; this table lets you override a value, add a flag, or blank one out
(blank = not set). **Restore Recommended** puts the default set back. The set
is stored in the project YAML under `engine_env:`, so a saved project records
exactly what its run had set, and the engine reads the same block at startup.

Change these only deliberately — for example, to reproduce an old run that
predates a fix.

### Body / HORIZONS

`Browse HORIZONS...` on the Simulation page searches JPL HORIZONS for a small
body, caches a `body_config.yaml`, and adds it to the body list; the **Body**
page then exposes its shape, orbit, spin, and surface properties — including
the six thermophysical fields that a built-in body cannot set. Selecting a
HORIZONS body also switches the albedo source to *None* and turns off primary
reflection, since an asteroid has no parent body to reflect from.

There is also an optional **Camera** tool (**View > Show Camera Tool**), hidden
by default, which drives `ptm_camera_sim` for synthetic imaging.

---

## 8. Confirming a setting took effect

Some engine settings apply only in certain modes or on certain body types —
the `coupling_face_fraction` case above, or thermophysical scalars on built-in
bodies. The GUI disables or labels the known cases; to confirm any setting
reached the run:

1. **Read the YAML preview.** Change the control and check that the key
   changed in the pane at the bottom of the configuration panel — the run uses
   exactly that file.
2. **Read the pre-flight lines.** Binary md5, output path, engine environment:
   a record of exactly what is about to run.
3. **Read the engine's banners in the console.** The engine announces what it
   did (`Radiation exchange enabled from config file`,
   `Loaded view factors from: ...`, `Face count: ...`) and warns when a set
   value is ignored. If a setting is never mentioned, check the YAML.
4. **Compare the outputs.** For anything that should change temperatures, run
   both arms and difference them, as [tutorial 04](04_crater_radiosity.md)
   does.

---

## 9. Troubleshooting

**"Could not find ptm_thermal_sim executable" / pre-flight error
`ptm_thermal_sim executable not found. Build the project first`.**
The GUI looks for `build/bin/ptm_thermal_sim` relative to the repository
(`gui/gui_core/simulation_runner.py:389-406`). You have not built the engine, or
you built it somewhere else. Follow [INSTALL.md](../../INSTALL.md):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTM_ENABLE_GPU=OFF \
      -DCSPICE_ROOT="$CSPICE_ROOT"
cmake --build build --target ptm_thermal_sim -j
```

**The run dies almost immediately with a SPICE error.**
The SPICE kernels are not downloaded (they are large, and not in the
repository):

```bash
python3 scripts/fetch_kernels.py --bodies moon    # Moon/Mercury core, ~44 MB
python3 scripts/fetch_kernels.py --bodies europa  # adds the Galilean ephemeris (~1.1 GB)
```

`data/kernels/*.tm` metakernels use paths **relative to the repository root**.
The GUI pins the working directory to the project root, so this works from the
GUI; it is a common failure when running the engine by hand from elsewhere.

**"Use GPU" is checked and the run fails or ignores it.**
CPU and GPU are a **build-time** choice. `PTM_ENABLE_GPU=OFF` is the default on
a fresh checkout, and PTM3D is designed to run well on ordinary CPUs using all
cores. For the GPU path, configure with `-DPTM_ENABLE_GPU=ON` and run the
one-time per-host calibration (`ptm_thermal_sim --calibrate-gpu`). Run the
calibration deliberately, not automatically: on some AMD hardware a wedged GPU
can take down the whole machine. On Apple Silicon (no OpenCL support), use the
CPU path.

**Everything in the console is red and the run "failed" but the outputs look
fine.** You are on an old build; severity is classified by content on current
`main`. Update.

**The launch is blocked with "Cloud placeholder (dataless) files".** Those files
have been evicted to the cloud; reading one hangs the engine indefinitely.
Hydrate them (Finder: right-click > "Always Keep on This Device", wait for the
download) or point the config at a local copy. Better: keep simulation input
data off cloud-synced folders entirely.

**No 3D view.** `pip install --upgrade pyvista pyvistaqt vtk`. Only the 3D tab
is affected; the rest of the app works without it.

**A control seems to do nothing.** See section 8, and check the console for an
engine warning saying the value is ignored.

---

## Regenerating the figures

The panel figures in this tutorial are rendered offscreen, straight from the GUI
object model:

```bash
python scripts/make_gui_tutorial_screenshots.py          # panels only
python scripts/make_gui_tutorial_screenshots.py --run    # + one tiny CPU run
```

It never opens a window on your display. The **whole window** and the
**3D/2D result views** cannot be captured this way: constructing
`VisualizationPanel` under the offscreen platform crashes inside headless
PyVista/VTK (SIGSEGV), and the main window constructs one. Those figures are
the `[SCREENSHOT-PAUL]` items above.

## Screenshots still to be taken on a live display

| # | What to set up | What to frame |
|---|---|---|
| 1 | Quit the GUI, run `defaults delete com.ptm3d.PTM3DSimulator bodyPickerShown`, relaunch | The first-launch "Select target body" dialog, body list visible |
| 2 | Fresh launch, Simulation page selected, nothing loaded | The whole window: toolbar, config panel, empty 3D view, console, status bar |
| 3 | Run the section-4 quickstart and catch it mid-run | Status bar with a partly-filled progress bar and the yellow "Running" badge, console streaming `Progress N/144` |
| 4 | Load `output/gui_tutorial_moon_quickstart`, 3D tab, Field = "Annual Max Temperature (K)", sub-solar hemisphere toward the camera | The 3D view plus the toolbar row (Timestep and Field selectors readable) |
| 5 | Same results, 2D Plots tab | The whole right-hand panel: temperature histogram and temperature-vs-latitude scatter |
