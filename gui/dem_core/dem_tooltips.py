"""
Tooltip strings for DEM/Mesh module parameters.
"""

TOOLTIPS = {
    # Input
    "input.dem_file": (
        "Path to the input DEM file (GeoTIFF .tif format).\n"
        "Supports equirectangular and polar stereographic projections.\n"
        "The file is read via GDAL."
    ),
    "input.body_radius_m": (
        "Mean radius of the planetary body in meters.\n"
        "Used for geographic-to-Cartesian coordinate conversion.\n"
        "Auto-detected from DEM metadata when available."
    ),

    # ROI
    "roi.enabled": "Enable circular region-of-interest extraction from the DEM.",
    "roi.center_lat": "Latitude of the ROI center in degrees.",
    "roi.center_lon": "Longitude of the ROI center in degrees.",
    "roi.radius_km": "Radius of the circular ROI in kilometers.",
    "roi.max_points": (
        "Maximum number of elevation points to retain in the ROI.\n"
        "The DEM is downsampled if needed to stay within this limit."
    ),

    # Conversion (Delaunay)
    "conversion.downsample": (
        "Downsampling factor for the DEM grid.\n"
        "1 = full resolution, 10 = every 10th pixel.\n"
        "Higher values produce smaller meshes faster."
    ),
    "conversion.roi_core_frac": (
        "Fraction of the ROI radius kept at full resolution.\n"
        "Default 0.15 = innermost 15% is not downsampled."
    ),
    "conversion.density_exp": (
        "Density falloff exponent for ROI point distribution.\n"
        "Controls how quickly point density decreases from ROI center.\n"
        "  1.0 = linear falloff (gentler)\n"
        "  2.0 = quadratic falloff (default, sharper transition)\n"
        "  0.5 = square root falloff (very gentle)\n"
        "Lower values keep more points at intermediate distances."
    ),
    "conversion.max_edge_m": (
        "Maximum triangle edge length in meters.\n"
        "  Auto (-1): compute threshold from point density (default)\n"
        "  Disable (0): no edge filtering (no holes in mesh)\n"
        "  Custom (>0): remove triangles with edges longer than this\n"
        "Edge filtering prevents spurious long triangles across gaps."
    ),
    "conversion.south_pole_coords": (
        "Apply south pole coordinate corrections:\n"
        "  - Swap Cartesian X/Y axes\n"
        "  - Orient X-axis right when 0 deg longitude is up\n"
        "  - Ensure negative Z at south pole\n"
        "Enable for south-polar DEMs (e.g., lunar south pole)."
    ),
    "conversion.swap_axes": (
        "Swap X/Y axes in polar stereographic projection.\n"
        "Use when the DEM projection has transposed easting/northing."
    ),
    "conversion.output_file": "Output path for the converted PLY mesh file.",

    # Quality Control
    "quality_control.enabled": (
        "Enable post-processing quality control sweep.\n"
        "Removes sliver triangles and fills small boundary holes."
    ),
    "quality_control.sliver_angle_deg": (
        "Minimum interior angle (degrees) for sliver removal.\n"
        "Triangles with any angle below this are removed.\n"
        "Default: 5.0 degrees."
    ),
    "quality_control.max_hole_edges": (
        "Maximum number of boundary loop edges to fill.\n"
        "Small holes (up to this many edges) are triangulated.\n"
        "0 = disable hole filling. Default: 20."
    ),
}
