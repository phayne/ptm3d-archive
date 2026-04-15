"""
Survey configuration: YAML loading, validation, and dataclass access.
"""
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta


@dataclass
class ROIConfig:
    center_lat: float = -84.5
    center_lon: float = 85.5
    radius_km: float = 75.0
    max_points: int = 2000000
    density_exp: float = 1.0
    core_frac: float = 0.10


@dataclass
class DEMConfig:
    file: str = ""
    south_pole_coords: bool = True


@dataclass
class TimeConfig:
    start: str = "2026-04-01T00:00:00"
    duration_years: float = 2.0
    lunation_days: float = 29.530588
    timesteps_per_lunation: int = 360

    @property
    def num_lunations(self) -> int:
        return int(self.duration_years * 365.25 / self.lunation_days) + 1

    @property
    def start_datetime(self) -> datetime:
        return datetime.fromisoformat(self.start)

    def lunation_start(self, index: int) -> str:
        dt = self.start_datetime + timedelta(days=index * self.lunation_days)
        return dt.strftime('%Y-%m-%dT%H:%M:%S')

    @property
    def hours_per_timestep(self) -> float:
        return self.lunation_days * 24.0 / self.timesteps_per_lunation


@dataclass
class IlluminationConfig:
    num_lunations: int = 25
    compute_duration_maps: bool = True
    include_primary_reflection: bool = True
    use_albedo_map: bool = False


@dataclass
class ThermalConfig:
    enabled: bool = True
    lunations: List[int] = field(default_factory=lambda: [7, 13])
    equilibration_orbits: float = 1.0
    radiation_exchange: bool = True
    max_coupling_iterations: int = 1
    use_gpu: bool = True
    view_factor_file: str = ""


@dataclass
class ThermophysicalConfig:
    albedo: float = 0.12
    emissivity: float = 0.95
    incidence_angle_albedo: bool = True
    ks: float = 0.00074
    kd: float = 0.0034
    H: float = 0.06
    rhos: float = 1100.0
    rhod: float = 1800.0
    cp0: float = 800.0
    Qb: float = 0.018
    chi: float = 2.7


@dataclass
class SelectionCriteria:
    min_illum_fraction: float = 0.0001
    max_illum_fraction: float = 0.20
    min_continuous_hours: float = 8.0
    max_temperature_K: float = 150.0
    min_cluster_size: int = 50


@dataclass
class NamedSite:
    name: str = ""
    lat: float = 0.0
    lon: float = 0.0


@dataclass
class ViewshedConfig:
    enabled: bool = False
    observer_height_m: float = 2.0
    max_range_km: float = 10.0


@dataclass
class TraverseConfig:
    enabled: bool = False
    start_lat: float = 0.0
    start_lon: float = 0.0
    start_time: str = "2026-10-01T00:00:00"
    max_distance_km: float = 50.0
    max_duration_days: float = 14.0
    return_to_start: bool = True
    min_sites: int = 3
    max_slope_deg: float = 15.0
    min_rest_hours: float = 2.0
    traverse_speed_kmh: float = 2.0
    require_illuminated: bool = True
    viewshed: ViewshedConfig = field(default_factory=ViewshedConfig)


@dataclass
class OutputConfig:
    base_dir: str = "output/survey"
    report_format: str = "html"
    clean_per_timestep_data: bool = True


@dataclass
class SurveyConfig:
    """Complete survey configuration."""
    name: str = "unnamed_survey"
    body: str = "Moon"
    body_radius: float = 1737400.0

    roi: ROIConfig = field(default_factory=ROIConfig)
    dem: DEMConfig = field(default_factory=DEMConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    thermophysical: ThermophysicalConfig = field(default_factory=ThermophysicalConfig)
    selection: SelectionCriteria = field(default_factory=SelectionCriteria)
    named_sites: List[NamedSite] = field(default_factory=list)
    traverse: TraverseConfig = field(default_factory=TraverseConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Derived paths
    @property
    def mesh_path(self) -> Path:
        return Path(self.output.base_dir) / "mesh.ply"

    @property
    def survey_dir(self) -> Path:
        return Path(self.output.base_dir) / "survey"

    @property
    def thermal_dir(self) -> Path:
        return Path(self.output.base_dir) / "thermal"

    @property
    def state_file(self) -> Path:
        return Path(self.output.base_dir) / "survey_state.json"

    @property
    def vf_path(self) -> Path:
        if self.thermal.view_factor_file:
            return Path(self.thermal.view_factor_file)
        return Path(self.output.base_dir) / "view_factors.hmat"

    @property
    def locations_csv(self) -> Path:
        return Path(self.output.base_dir) / "target_locations.csv"

    @property
    def figures_dir(self) -> Path:
        return Path(self.output.base_dir) / "figures"

    def validate(self) -> List[str]:
        """Return list of validation errors (empty if valid)."""
        errors = []
        if not self.dem.file:
            errors.append("dem.file is required")
        elif not Path(self.dem.file).exists():
            errors.append(f"DEM file not found: {self.dem.file}")
        if self.roi.radius_km <= 0:
            errors.append("roi.radius_km must be positive")
        if self.time.timesteps_per_lunation < 10:
            errors.append("time.timesteps_per_lunation must be >= 10")
        if self.illumination.num_lunations < 1:
            errors.append("illumination.num_lunations must be >= 1")
        if self.thermal.enabled and not self.thermal.lunations:
            errors.append("thermal.lunations must be specified when thermal.enabled")
        if self.traverse.enabled:
            if self.traverse.max_distance_km <= 0:
                errors.append("traverse.max_distance_km must be positive")
            if self.traverse.max_slope_deg <= 0:
                errors.append("traverse.max_slope_deg must be positive")
        return errors


def load_config(path: str) -> SurveyConfig:
    """Load a SurveyConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = SurveyConfig()

    # Top-level survey fields
    survey = raw.get('survey', {})
    cfg.name = survey.get('name', cfg.name)
    cfg.body = survey.get('body', cfg.body)
    cfg.body_radius = survey.get('body_radius', cfg.body_radius)

    # ROI
    roi = raw.get('roi', {})
    cfg.roi = ROIConfig(**{k: roi[k] for k in roi if hasattr(cfg.roi, k)})

    # DEM
    dem = raw.get('dem', {})
    cfg.dem = DEMConfig(**{k: dem[k] for k in dem if hasattr(cfg.dem, k)})

    # Time
    time_cfg = raw.get('time', {})
    cfg.time = TimeConfig(**{k: time_cfg[k] for k in time_cfg if hasattr(cfg.time, k)})

    # Illumination
    illum = raw.get('illumination', {})
    cfg.illumination = IlluminationConfig(**{k: illum[k] for k in illum if hasattr(cfg.illumination, k)})

    # Thermal
    therm = raw.get('thermal', {})
    cfg.thermal = ThermalConfig(**{k: therm[k] for k in therm if hasattr(cfg.thermal, k)})

    # Thermophysical
    tp = raw.get('thermophysical', {})
    cfg.thermophysical = ThermophysicalConfig(**{k: tp[k] for k in tp if hasattr(cfg.thermophysical, k)})

    # Selection criteria
    sel = raw.get('selection_criteria', {})
    cfg.selection = SelectionCriteria(**{k: sel[k] for k in sel if hasattr(cfg.selection, k)})

    # Named sites
    for site in raw.get('named_sites', []):
        cfg.named_sites.append(NamedSite(**site))

    # Traverse
    trav = raw.get('traverse', {})
    vs = trav.pop('viewshed', {})
    cfg.traverse = TraverseConfig(**{k: trav[k] for k in trav if hasattr(cfg.traverse, k)})
    if vs:
        cfg.traverse.viewshed = ViewshedConfig(**{k: vs[k] for k in vs if hasattr(cfg.traverse.viewshed, k)})

    # Output
    out = raw.get('output', {})
    cfg.output = OutputConfig(**{k: out[k] for k in out if hasattr(cfg.output, k)})

    return cfg


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python survey_config.py <config.yaml>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    errors = cfg.validate()
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"Survey: {cfg.name}")
    print(f"Body: {cfg.body} (R={cfg.body_radius/1000:.1f} km)")
    print(f"ROI: ({cfg.roi.center_lat}, {cfg.roi.center_lon}), r={cfg.roi.radius_km} km")
    print(f"Time: {cfg.time.start}, {cfg.illumination.num_lunations} lunations")
    print(f"Thermal: {'enabled' if cfg.thermal.enabled else 'disabled'} (lunations {cfg.thermal.lunations})")
    print(f"Traverse: {'enabled' if cfg.traverse.enabled else 'disabled'}")
    print(f"Output: {cfg.output.base_dir}")
    print("Configuration valid.")
