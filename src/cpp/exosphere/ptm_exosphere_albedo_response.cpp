// --albedo-response diurnal: build the per-face diurnal-shape temperature response
// (diurnal_albedo_response.h) from the thermal run. Kept in its own translation
// unit so the legacy code in ptm_exosphere_integration.cpp is compiled unchanged.

#include "ptm_exosphere_integration.h"
#include "../fileio/frame_naming.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace ptm {
namespace exosphere {

namespace {

// One per-step thermal output frame (uint32 count + float32[count]).
bool readFrame(const std::string& dir, const std::string& component, int t, int num_timesteps,
               size_t n_faces, std::vector<float>& out, std::string& err) {
    const std::string path = ptm::resolveFrameFile(dir + "/" + component, component, t);
    std::ifstream in(path, std::ios::binary);
    if (path.empty() || !in) {
        err = "missing " + component + " frame for step " + std::to_string(t) + " under " + dir + "/" +
              component + "/ (tried " + component + "_" + ptm::frameIndexString(t, num_timesteps) +
              ".bin and the legacy 4-digit name)";
        return false;
    }
    uint32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in || n != n_faces) {
        err = path + " holds " + std::to_string(n) + " values; the mesh has " + std::to_string(n_faces) + " faces";
        return false;
    }
    out.resize(n_faces);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n_faces * sizeof(float)));
    if (!in) {
        err = "short read on " + path;
        return false;
    }
    return true;
}

// The thermal run's output epochs (time_steps.txt: elapsed seconds per row). The
// response assumes nt samples uniformly spaced over exactly one forcing period.
bool checkOneUniformPeriod(const std::string& dir, int nt, double period_s, std::string& err) {
    std::ifstream in(dir + "/time_steps.txt");
    if (!in) {
        err = "no " + dir + "/time_steps.txt to verify that the output window is one forcing period";
        return false;
    }
    std::vector<double> t;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double v;
        if (ss >> v) t.push_back(v);
    }
    if (static_cast<int>(t.size()) != nt) {
        err = "time_steps.txt lists " + std::to_string(t.size()) + " epochs; " + std::to_string(nt) + " were loaded";
        return false;
    }
    const double step = period_s / static_cast<double>(nt);
    for (int i = 1; i < nt; ++i) {
        if (std::abs((t[i] - t[i - 1]) - step) > 1e-6 * step) {
            std::ostringstream o;
            o << std::setprecision(10) << "output epochs are not uniform at period/" << nt << " = " << step
              << " s (step " << i << ": " << (t[i] - t[i - 1]) << " s); the diurnal response needs one "
              << "periodic window";
            err = o.str();
            return false;
        }
    }
    return true;
}

}  // namespace

bool PTMExosphereIntegration::prepareDiurnalAlbedoResponse(const ThermalColumn& column,
                                                           const std::string& thermal_dir,
                                                           const DiurnalResponseOptions& options) {
    const auto t_start = std::chrono::steady_clock::now();
    const std::string col_err = column.validate();
    if (!col_err.empty()) {
        std::cerr << "Error [albedo-response]: thermal column: " << col_err << std::endl;
        return false;
    }
    if (!albedo_coupling_enabled_ || !albedo_refresh_.seeded()) {
        std::cerr << "Error [albedo-response]: enable albedo coupling before the diurnal response" << std::endl;
        return false;
    }
    if (albedo_refreshes_fired_ > 0) {
        std::cerr << "Error [albedo-response]: the temperatures were already refreshed; the response must be "
                  << "built from the thermal run's own temperatures" << std::endl;
        return false;
    }
    const size_t nf = static_cast<size_t>(num_faces_);
    if (surface_temperatures_.size() != nf || surface_temperatures_.empty() || surface_temperatures_[0].empty()) {
        std::cerr << "Error [albedo-response]: no temperatures loaded" << std::endl;
        return false;
    }
    const int nt = static_cast<int>(surface_temperatures_[0].size());
    std::string err;
    if (!checkOneUniformPeriod(thermal_dir, nt, column.period_s, err)) {
        std::cerr << "Error [albedo-response]: " << err << std::endl;
        return false;
    }

    // The absorbed solar flux is rebuilt as (1 - A_thermal) x illumination, so the
    // declared thermal-run albedo must be the albedo the run's surface balance used:
    // statistics/effective_albedo.bin (flux-weighted operative albedo, NaN where never
    // lit). A wrong --thermal-albedo(-map) or an incidence-dependent albedo law differs.
    {
        const std::string path = thermal_dir + "/statistics/effective_albedo.bin";
        std::ifstream in(path, std::ios::binary);
        uint32_t n = 0;
        if (in) in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!in || n != nf) {
            std::cerr << "Error [albedo-response]: cannot read " << path << " (" << nf << " faces) to verify "
                      << "the declared thermal-run albedo" << std::endl;
            return false;
        }
        std::vector<float> eff(nf);
        in.read(reinterpret_cast<char*>(eff.data()), static_cast<std::streamsize>(nf * sizeof(float)));
        if (!in) {
            std::cerr << "Error [albedo-response]: short read on " << path << std::endl;
            return false;
        }
        size_t n_over = 0, worst = 0, n_lit = 0;
        double dmax = 0.0;
        for (size_t f = 0; f < nf; ++f) {
            if (!std::isfinite(eff[f])) continue;   // never lit: no absorbed solar flux to rebuild
            ++n_lit;
            const double d = std::abs(static_cast<double>(eff[f]) - thermal_face_albedos_[f]);
            if (d > 1e-4) ++n_over;
            if (d > dmax) { dmax = d; worst = f; }
        }
        std::cout << "  Thermal-run albedo check: declared (" << thermal_albedo_source_ << ") vs the run's "
                  << "operative albedo on " << n_lit << " lit faces: max |d| " << std::scientific
                  << std::setprecision(3) << dmax << std::defaultfloat << " (limit 1e-4)" << std::endl;
        if (n_over > 0) {
            std::cerr << "Error [albedo-response]: the declared thermal-run albedo differs from the albedo the "
                      << "thermal run absorbed with by more than 1e-4 on " << n_over << " lit faces (worst face "
                      << worst << ": declared " << thermal_face_albedos_[worst] << ", " << path << " "
                      << eff[worst] << "). Fix --thermal-albedo / --thermal-albedo-map; the legacy law would "
                      << "silently accept it." << std::endl;
            return false;
        }
    }

    // Absorbed solar (albedo-weighted) and absorbed infrared, face-major.
    const size_t ntu = static_cast<size_t>(nt);
    std::vector<float> Fs(nf * ntu), Qir(nf * ntu), frame;
    for (int t = 0; t < nt; ++t) {
        if (!readFrame(thermal_dir, "illumination", t, nt, nf, frame, err)) {
            std::cerr << "Error [albedo-response]: " << err << std::endl;
            return false;
        }
        for (size_t f = 0; f < nf; ++f) Fs[f * ntu + static_cast<size_t>(t)] = frame[f];
        if (!readFrame(thermal_dir, "primary_ir", t, nt, nf, frame, err)) {
            std::cerr << "Error [albedo-response]: " << err << std::endl;
            return false;
        }
        for (size_t f = 0; f < nf; ++f)
            Qir[f * ntu + static_cast<size_t>(t)] = static_cast<float>(column.emissivity * frame[f]);
    }

    diurnal_reference_albedo_ = albedo_refresh_.baseline();
    diurnal_T0_.assign(nf * ntu, 0.0f);
    diurnal_g1_.assign(nf * ntu, 0.0f);
    diurnal_g2_.assign(nf * ntu, 0.0f);
    diurnal_fallback_.assign(nf, 0);
    std::vector<float> balance(nf, 0.0f);
    std::vector<char> warm(nf, 0);
    size_t n_bad_albedo = 0, n_failed = 0;

    #pragma omp parallel reduction(+:n_bad_albedo, n_failed)
    {
        DiurnalResponseWorkspace ws;
        std::vector<double> T0(ntu), fs(ntu), qi(ntu), s1(ntu), s2(ntu);
        #pragma omp for schedule(dynamic, 64)
        for (size_t f = 0; f < nf; ++f) {
            double tmax = 0.0;
            for (size_t t = 0; t < ntu; ++t) {
                T0[t] = surface_temperatures_[f][t];
                diurnal_T0_[f * ntu + t] = static_cast<float>(T0[t]);
                tmax = std::max(tmax, T0[t]);
            }
            warm[f] = tmax > 110.0 ? 1 : 0;
            const double a_th = thermal_face_albedos_[f];
            if (!std::isfinite(a_th) || a_th < 0.0 || a_th >= 1.0) {
                // What the thermal run absorbed is unknown here: no response.
                diurnal_fallback_[f] = 1;
                ++n_bad_albedo;
                continue;
            }
            for (size_t t = 0; t < ntu; ++t) {
                fs[t] = (1.0 - a_th) * static_cast<double>(Fs[f * ntu + t]);
                qi[t] = static_cast<double>(Qir[f * ntu + t]);
            }
            DiurnalResponseFaceInfo inf;
            if (!computeDiurnalAlbedoResponse(column, T0.data(), fs.data(), qi.data(), nt, options, ws,
                                              s1.data(), s2.data(), &inf)) {
                diurnal_fallback_[f] = 1;
                ++n_failed;
                continue;
            }
            diurnalLogCoefficients(T0.data(), s1.data(), s2.data(), nt, &diurnal_g1_[f * ntu], &diurnal_g2_[f * ntu]);
            balance[f] = static_cast<float>(inf.balance_rms / std::max(1e-30, inf.balance_scale));
        }
    }

    // Energy-balance check: does this column reproduce the thermal run?
    std::vector<float> b_all, b_warm;
    b_all.reserve(nf);
    for (size_t f = 0; f < nf; ++f) {
        if (diurnal_fallback_[f]) continue;
        b_all.push_back(balance[f]);
        if (warm[f]) b_warm.push_back(balance[f]);
    }
    auto quant = [](std::vector<float> v, double q) -> double {
        if (v.empty()) return 0.0;
        const size_t k = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size() - 1)));
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
        return v[k];
    };
    const double b_med = quant(b_all, 0.5), b_p99 = quant(b_all, 0.99);
    const double bw_max = b_warm.empty() ? 0.0 : *std::max_element(b_warm.begin(), b_warm.end());
    double g1min = 1e300, g1max = -1e300;
    for (size_t f = 0; f < nf; ++f) {
        if (diurnal_fallback_[f]) continue;
        for (size_t t = 0; t < ntu; ++t) {
            g1min = std::min(g1min, static_cast<double>(diurnal_g1_[f * ntu + t]));
            g1max = std::max(g1max, static_cast<double>(diurnal_g1_[f * ntu + t]));
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    std::ostringstream rep;
    rep << std::defaultfloat << std::setprecision(4);
    rep << "  Temperature response: diurnal shape (--albedo-response diurnal): per-face second-order response of "
        << column.layers() << "-layer column, " << options.substeps << " substeps, " << options.base_cycles
        << " base cycles; built in " << std::fixed << std::setprecision(1) << seconds << " s\n"
        << std::defaultfloat << std::setprecision(4)
        << "    g1 = dlnT/dln(1-A) range [" << g1min << ", " << g1max << "] (radiative equilibrium 0.25)\n"
        << "    Energy balance of the thermal run on this column (rms / flux scale): median " << b_med
        << ", p99 " << b_p99 << ", max on T_max > 110 K faces " << bw_max << " (limit: median 0.015)\n"
        << "    Faces on the 1/4-power law (no response): " << (n_bad_albedo + n_failed) << " (thermal albedo "
        << "outside [0, 1): " << n_bad_albedo << "; periodic solve failed: " << n_failed << ")\n";
    diurnal_response_report_ = rep.str();
    std::cout << "\nAlbedo response:\n" << diurnal_response_report_ << std::flush;

    if (!(b_med <= 0.015)) {
        std::cerr << "Error [albedo-response]: the thermal column does not reproduce the thermal run's surface "
                  << "energy balance (median residual " << b_med << " of the flux scale > 0.015). Check first "
                  << "that the declared thermal-run albedo (" << thermal_albedo_source_ << ") is the one the "
                  << "thermal run used: a wrong value fails here, and the legacy law would silently accept "
                  << "it. Otherwise the run's grid, conductivity, heat capacity or emissivity differ from the "
                  << "column built here, and only --albedo-response rescale can use it." << std::endl;
        return false;
    }
    if (n_bad_albedo + n_failed > 0) {
        std::cerr << "WARNING [albedo-response]: " << (n_bad_albedo + n_failed) << " faces have no diurnal "
                  << "response and use the 1/4-power law from their t = 0 temperatures (" << n_bad_albedo
                  << " with a thermal albedo outside [0, 1), " << n_failed << " failed periodic solves)"
                  << std::endl;
    }
    diurnal_albedo_response_ = true;
    return true;
}

}  // namespace exosphere
}  // namespace ptm
