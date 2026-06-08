/**
 * @file horizon_field.h
 * @brief Heightfield-horizon visibility for view-factor occlusion, robust to the
 *        grazing-tangent geometry where a single Embree centroid ray over-occludes
 *        (floor->rim coupling collapse on rough fine meshes). Validated on the
 *        analytic bowl harness: recovers ~6x more rim coupling than the thin ray
 *        (see src/python/bowl_estimator_test.py and memory floor_grazing_vf_degradation).
 *
 * Method: rasterize the surface to a height grid in a local up-frame (max height
 * per horizontal cell), then test visibility(i->j) by the DISTANCE-AWARE running
 * maximum terrain elevation angle along the azimuth: j is occluded iff some terrain
 * nearer than j rises above the i->j sightline angle. This matches the DEM horizon
 * scan (the geometric ground truth) and is far less fragile than a thin triangle ray.
 *
 * Header-only; double precision; no Embree/Eigen dependency.
 */
#ifndef PTM_HORIZON_FIELD_H
#define PTM_HORIZON_FIELD_H

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace radiation {

class HorizonField {
public:
    // pts: per-face centers in ANY consistent frame (e.g. the H-matrix recentered
    //      face_centers_). up: local "up" unit vector (radial for a planetary patch).
    // cell: horizontal grid cell size (m). Use ~ the median face pitch.
    HorizonField() = default;

    void build(const std::vector<std::array<double, 3>>& pts,
               const std::array<double, 3>& up, double cell) {
        up_ = normalize(up);
        // two horizontal basis vectors orthogonal to up
        std::array<double, 3> ref = (std::fabs(up_[2]) < 0.9)
            ? std::array<double, 3>{{0, 0, 1}} : std::array<double, 3>{{1, 0, 0}};
        e_ = normalize(cross(up_, ref));
        n_ = cross(up_, e_);
        cell_ = (cell > 0) ? cell : 1.0;
        inv_cell_ = 1.0 / cell_;

        const size_t N = pts.size();
        double emin = 1e300, emax = -1e300, nmin = 1e300, nmax = -1e300;
        eproj_.resize(N); nproj_.resize(N); uproj_.resize(N);
        for (size_t i = 0; i < N; ++i) {
            double ex = dot(pts[i], e_), nx = dot(pts[i], n_), ux = dot(pts[i], up_);
            eproj_[i] = ex; nproj_[i] = nx; uproj_[i] = ux;
            emin = std::min(emin, ex); emax = std::max(emax, ex);
            nmin = std::min(nmin, nx); nmax = std::max(nmax, nx);
        }
        emin_ = emin; nmin_ = nmin;
        ne_ = static_cast<int>((emax - emin) * inv_cell_) + 2;
        nn_ = static_cast<int>((nmax - nmin) * inv_cell_) + 2;
        // guard against pathological sizes
        if ((long long)ne_ * nn_ > 200000000LL) {  // > ~1.6 GB of double; coarsen
            double scale = std::sqrt((double)ne_ * nn_ / 1.5e8);
            cell_ *= scale; inv_cell_ = 1.0 / cell_;
            ne_ = static_cast<int>((emax - emin) * inv_cell_) + 2;
            nn_ = static_cast<int>((nmax - nmin) * inv_cell_) + 2;
        }
        hgrid_.assign((size_t)ne_ * nn_, -1e300);
        for (size_t i = 0; i < N; ++i) {
            int ie = (int)((eproj_[i] - emin_) * inv_cell_);
            int in = (int)((nproj_[i] - nmin_) * inv_cell_);
            double& h = hgrid_[(size_t)ie * nn_ + in];
            if (uproj_[i] > h) h = uproj_[i];
        }
        built_ = true;
    }

    bool built() const { return built_; }

    // Is face j occluded from face i? (distance-aware running-max horizon test.)
    // Indices into the same pts array used in build().
    bool occluded(int i, int j) const {
        const double ei = eproj_[i], ni = nproj_[i], ui = uproj_[i];
        const double dej = eproj_[j] - ei, dnj = nproj_[j] - ni, duj = uproj_[j] - ui;
        const double hd = std::sqrt(dej * dej + dnj * dnj);
        if (hd < cell_) return false;                       // same cell: treat visible
        const double theta_j = std::atan2(duj, hd);         // sightline elevation to j
        const double dirx = dej / hd, diry = dnj / hd;      // horizontal march direction
        // march from i toward j over the grid, tracking max terrain angle BEFORE j
        const double step = cell_;
        const double margin = 0.5 * cell_;                  // stop just short of j
        double running_max = -M_PI;
        for (double s = step; s < hd - margin; s += step) {
            double ex = ei + s * dirx, nx = ni + s * diry;
            int ie = (int)((ex - emin_) * inv_cell_);
            int in = (int)((nx - nmin_) * inv_cell_);
            if (ie < 0 || ie >= ne_ || in < 0 || in >= nn_) continue;
            double h = hgrid_[(size_t)ie * nn_ + in];
            if (h < -1e299) continue;                        // empty cell
            double ang = std::atan2(h - ui, s);
            if (ang > running_max) running_max = ang;
        }
        // occluded iff nearer terrain rises above the sightline (with a small slack)
        static const double slack = 0.25 * M_PI / 180.0;
        return running_max > theta_j + slack;
    }

private:
    static double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    static std::array<double, 3> cross(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
    }
    static std::array<double, 3> normalize(const std::array<double, 3>& a) {
        double L = std::sqrt(dot(a, a)); return (L > 0) ? std::array<double, 3>{{a[0] / L, a[1] / L, a[2] / L}} : a;
    }

    bool built_ = false;
    std::array<double, 3> up_{{0, 0, 1}}, e_{{1, 0, 0}}, n_{{0, 1, 0}};
    double cell_ = 1.0, inv_cell_ = 1.0, emin_ = 0, nmin_ = 0;
    int ne_ = 0, nn_ = 0;
    std::vector<double> hgrid_;
    std::vector<double> eproj_, nproj_, uproj_;
};

}  // namespace radiation

#endif  // PTM_HORIZON_FIELD_H
