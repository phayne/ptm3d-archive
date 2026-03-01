"""
Tooltips for Viewshed GUI configuration parameters.
"""

TOOLTIPS = {
    # Input
    "input.ptm_output_dir": (
        "Path to the PTM3D output directory containing mesh.ply, "
        "temperature/, statistics/, and time_steps.txt."
    ),
    "input.mesh_file": (
        "Path to the mesh PLY file. If empty, defaults to {ptm_output_dir}/mesh.ply."
    ),
    "input.waypoints_csv": (
        "CSV file with waypoint positions. Expected columns: "
        "label, latitude, longitude, height."
    ),

    # Mission
    "mission.start_utc": (
        "Mission start date in ISO format (YYYY-MM-DD). "
        "Waypoints are spaced evenly between start and end dates."
    ),
    "mission.end_utc": (
        "Mission end date in ISO format (YYYY-MM-DD)."
    ),
    "mission.body_radius_m": (
        "Radius of the planetary body in metres. "
        "Default: 1,737,400 m (Moon)."
    ),

    # Viewshed
    "viewshed.max_emission_angle_deg": (
        "Maximum emission angle (observer-to-face) in degrees. "
        "Faces with larger emission angles are not considered visible. "
        "Default: 85 degrees."
    ),
    "viewshed.max_range_km": (
        "Maximum range from observer to face centre in km. "
        "Faces beyond this distance are excluded. Default: 100 km."
    ),

    # Ice distribution
    "ice.temperature_threshold_K": (
        "Temperature threshold for ice stability. Faces with T_max below "
        "this value are assigned the specified ice fraction. Default: 110 K."
    ),
    "ice.ice_fraction": (
        "Mass fraction of water ice assigned to thermally stable faces. "
        "Default: 0.20 (20%)."
    ),
    "ice.gradient_scale_K": (
        "If > 0, creates a smooth ice-fraction gradient near the temperature "
        "threshold instead of a binary step. The value sets the width of the "
        "transition in Kelvin. Default: 0 (binary on/off)."
    ),

    # Output
    "output.output_dir": (
        "Directory where output .bin maps and summary files are written."
    ),
    "output.write_truth_map": (
        "Write the truth ice fraction map (truth_ice_fraction.bin)."
    ),
    "output.write_per_waypoint": (
        "Write per-waypoint output maps (visibility, distance, emission angle, "
        "SNR, estimated ice fraction) in addition to combined best-estimate maps."
    ),

    # Masks
    "masks.field": (
        "The data field to evaluate. Static fields (T_max, latitude, etc.) "
        "are computed once; per-waypoint fields require selecting a waypoint."
    ),
    "masks.operator": (
        "Comparison operator: <, <=, ==, >=, >, !="
    ),
    "masks.value": (
        "Threshold value for the comparison."
    ),
    "masks.connector": (
        "Logical connector to the next condition: AND or OR."
    ),

    # Aggregation
    "aggregation.visibility_count": (
        "For each face, count how many waypoints it is visible from. "
        "Output: visibility_count.bin"
    ),
    "aggregation.illumination_hours": (
        "For each face, total hours of direct solar illumination during "
        "the traverse window. Reads PTM3D illumination data or uses a "
        "temperature-based proxy."
    ),
    "aggregation.mean_temperature": (
        "Average surface temperature across all timesteps in the traverse "
        "window. Output: temperature_mean_traverse.bin"
    ),
    "aggregation.earth_visibility_fraction": (
        "For each face, the fraction of traverse-window timesteps when the "
        "primary body (Earth) is above the horizon. Derived from PTM3D "
        "primary IR flux: a face is Earth-visible when primary_ir > 0. "
        "Output: earth_visibility_fraction.bin"
    ),

    # Output format options
    "output.write_vtk": (
        "Export viewshed results as a time sequence of VTU files "
        "(one per waypoint) with a PVD collection file for ParaView. "
        "Each VTU contains per-waypoint fields (visibility, distance, "
        "emission angle, elevation angle, temperature, solar flux) and "
        "aggregation fields (visibility count, illumination hours, etc.). "
        "Open the .pvd file in ParaView to animate through waypoints. "
        "Requires pyvista."
    ),
    "output.write_hdf5": (
        "Export all output fields and mesh data as a single HDF5 file "
        "(.h5). Compact, portable format for further analysis. Requires h5py."
    ),
}


def get_tooltip(key):
    """Return tooltip text for a configuration key, or empty string."""
    return TOOLTIPS.get(key, "")
