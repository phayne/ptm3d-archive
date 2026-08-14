#ifndef ALBEDO_REGISTRATION_CHECK_H
#define ALBEDO_REGISTRATION_CHECK_H

#include <string>
#include <vector>

namespace ptm {

/**
 * @brief Result of a per-face albedo map registration check.
 *
 * `applicable` is false whenever no verdict was possible — either the body has
 * no published dark/bright reference signature, or the mesh does not cover
 * enough of the reference sites (a regional DEM mesh, for example). In that
 * case every other field is meaningless and nothing is reported.
 */
struct AlbedoRegistrationReport {
    bool applicable = false;   ///< a verdict was actually reached
    bool suspect = false;      ///< the map reads inverted / unregistered
    double dark_mean = 0.0;    ///< mean albedo sampled at the DARK reference sites
    double bright_mean = 0.0;  ///< mean albedo sampled at the BRIGHT reference sites
    double contrast = 0.0;     ///< bright_mean - dark_mean; must be positive
    int dark_n = 0;            ///< dark reference sites actually resolved by the mesh
    int bright_n = 0;          ///< bright reference sites actually resolved by the mesh

    // Latitude (hemisphere) axis, from northern maria vs their southern mirrors
    // at matched longitude. Only meaningful when ns_applicable is true.
    bool ns_applicable = false;
    bool latitude_suspect = false;   ///< hemispheres appear swapped
    double north_mare_mean = 0.0;
    double south_mirror_mean = 0.0;
    double ns_contrast = 0.0;        ///< south_mirror_mean - north_mare_mean; must be positive
};

/**
 * @brief Detect (and disclose) a longitude/latitude-misregistered albedo map.
 *
 * A per-face albedo map that has been rolled in longitude — the PTM-34 defect,
 * where the lunar Bond-albedo FITS was mislabeled by 180 degrees — still has the
 * correct VALUE DISTRIBUTION, so the mean albedo, the albedo range and the
 * effective absorbed albedo all look completely normal. Only the SPATIAL
 * registration is wrong. This check is therefore spatial: it samples the map at
 * sites whose albedo is known from the literature and asserts the physical
 * ordering (lunar maria are dark, ~0.07-0.09 Bond; highlands are bright, ~0.15-0.17).
 *
 * The check is DETECT-AND-DISCLOSE. It never modifies the map and never aborts
 * the run: a misregistered map is a data problem the user must fix upstream, and
 * silently rolling it here would break any body whose map is already correct.
 *
 * Scope is deliberately narrow, so it cannot fire spuriously:
 *   - Only bodies with a published reference signature are checked (currently
 *     the Moon). Europa, Mars and every generic body return applicable=false
 *     and produce no output. Europa's FITS registers correctly through the same
 *     loader, so a blanket correction there would be a regression.
 *   - Every reference site must be resolved by a face within
 *     `kMaxAnchorSeparationDeg` of it. On a regional DEM mesh the far-side sites
 *     are not present at all, so the check declines to render a verdict rather
 *     than reporting a false positive off the nearest few faces.
 *
 * @param body_name     Body name as resolved from the config (e.g. "Moon").
 * @param face_lat_deg  Per-face latitude, degrees north-positive.
 * @param face_lon_deg  Per-face east longitude, degrees (any branch; normalized here).
 * @param face_albedos  Per-face Bond albedo, same ordering as the lat/lon arrays.
 * @param source_label  Path or description of the map, quoted back in the warning.
 * @return the report; also prints a warning to stderr when `suspect` is true.
 */
AlbedoRegistrationReport checkAlbedoMapRegistration(
    const std::string& body_name,
    const std::vector<double>& face_lat_deg,
    const std::vector<double>& face_lon_deg,
    const std::vector<double>& face_albedos,
    const std::string& source_label);

/// Minimum acceptable (bright - dark) contrast. Measured on this project's maps:
/// correctly registered lunar bins give +0.0775 (res3) to +0.0797 (res9); the
/// 180-degree-rolled bins give -0.0241 to -0.0495. The gate sits far below the
/// correct value and far above the broken one.
constexpr double kMinAlbedoContrast = 0.02;

/// A reference site counts as resolved only if some face center lies within this
/// angular distance of it. Measured worst-case nearest-face separation on the
/// global lunar icospheres: 4.94 deg at res3, 0.08 deg at res9.
constexpr double kMaxAnchorSeparationDeg = 6.0;

/// Reference sites needed on each side before a verdict is rendered.
constexpr int kMinResolvedAnchors = 3;

}  // namespace ptm

#endif  // ALBEDO_REGISTRATION_CHECK_H
