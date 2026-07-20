#ifndef BODY_SHAPE_H
#define BODY_SHAPE_H

#include <string>
#include <cmath>

namespace orbital {

// Shape descriptor for a planetary body
struct BodyShape {
    enum Type {
        SPHERICAL,       // Uniform sphere: uses radius_km
        ELLIPSOIDAL,     // Tri-axial ellipsoid: uses semi_axis_a/b/c_km
        CUSTOM_MESH      // User-provided mesh file
    };

    Type type = SPHERICAL;

    // Spherical parameters
    double radius_km = 100.0;

    // Ellipsoidal parameters (semi-axes in km)
    // a = equatorial x-axis, b = equatorial y-axis, c = polar axis
    double semi_axis_a_km = 0.0;
    double semi_axis_b_km = 0.0;
    double semi_axis_c_km = 0.0;

    // Custom mesh parameters
    std::string custom_mesh_file;   // Path to OBJ/PLY/VTK file
    double mesh_scale = 1.0;        // Scale factor applied to mesh coordinates

    // Compute mean radius (km) for thermal skin depth scaling
    double meanRadiusKm() const {
        switch (type) {
            case SPHERICAL:
                return radius_km;
            case ELLIPSOIDAL:
                // Geometric mean of semi-axes
                return std::cbrt(semi_axis_a_km * semi_axis_b_km * semi_axis_c_km);
            case CUSTOM_MESH:
                return radius_km;  // User should set this for custom meshes
        }
        return radius_km;
    }

    // Compute mean radius in meters
    double meanRadiusMeters() const {
        return meanRadiusKm() * 1000.0;
    }

    // Validate shape parameters
    bool isValid() const {
        switch (type) {
            case SPHERICAL:
                return radius_km > 0.0;
            case ELLIPSOIDAL:
                return semi_axis_a_km > 0.0 && semi_axis_b_km > 0.0 && semi_axis_c_km > 0.0;
            case CUSTOM_MESH:
                return !custom_mesh_file.empty();
        }
        return false;
    }
};

} // namespace orbital

#endif // BODY_SHAPE_H
