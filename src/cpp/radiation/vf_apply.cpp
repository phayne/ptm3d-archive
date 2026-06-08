/**
 * @file vf_apply.cpp
 * @brief Compute y = F * v for a saved H-matrix view-factor matrix and an
 *        arbitrary per-face input vector v (float64 .npy). Diagnostic for
 *        WHERE a face's view-factor coupling points (e.g. floor -> warm rim):
 *        feed v_j = 1{warm}  -> y_i = floor i's VF to the warm set, or
 *        v_j = sigma*T_j^4   -> y_i = single-bounce incident IR on face i.
 *
 * Usage: vf_apply <hmatrix.hmat> <in_v.npy float64> <out_y.npy>
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "hmatrix_view_factor.h"

namespace {
void write_npy(const std::string& path, const std::vector<double>& data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    std::string hdr = "{'descr': '<f8', 'fortran_order': False, 'shape': (" +
                      std::to_string(n) + ",), }";
    size_t total = 10 + hdr.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    hdr.append(pad, ' ');
    hdr += '\n';
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    const char magic[8] = {'\x93', 'N', 'U', 'M', 'P', 'Y', '\x01', '\x00'};
    f.write(magic, 8);
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(hdr.data(), hdr.size());
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(double)));
}

// Minimal .npy float64 1-D reader (assumes <f8, C order, 1-D; handles v1.0 header).
std::vector<double> read_npy_f8(const std::string& path, size_t expect_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << path << "\n"; return {}; }
    char magic[8];
    f.read(magic, 8);
    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string hdr(hlen, '\0');
    f.read(&hdr[0], hlen);
    if (hdr.find("<f8") == std::string::npos)
        std::cerr << "WARNING: " << path << " not <f8; got: " << hdr << "\n";
    std::vector<double> v(expect_n);
    f.read(reinterpret_cast<char*>(v.data()),
           static_cast<std::streamsize>(expect_n * sizeof(double)));
    if (!f) std::cerr << "WARNING: short read on " << path << "\n";
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: vf_apply <hmatrix.hmat> <in_v.npy> <out_y.npy>\n";
        return 2;
    }
    radiation::HMatrixViewFactor H;
    std::cout << "Loading H-matrix: " << argv[1] << std::endl;
    if (!H.readBinaryMmap(argv[1]) && !H.readBinary(argv[1])) {
        std::cerr << "ERROR loading H-matrix\n"; return 1;
    }
    size_t N = H.numRows();
    std::cout << "  N = " << N << " faces" << std::endl;

    std::vector<double> v = read_npy_f8(argv[2], N);
    if (v.size() != N) { std::cerr << "ERROR: v size mismatch\n"; return 1; }

    std::vector<double> y(N, 0.0);
    H.matvec(v, y);

    double mn = 1e30, mx = -1e30, sum = 0.0;
    for (double r : y) { mn = std::min(mn, r); mx = std::max(mx, r); sum += r; }
    std::cout << "  y = F*v : min=" << mn << " mean=" << (sum / std::max<size_t>(1, N))
              << " max=" << mx << std::endl;
    write_npy(argv[3], y, N);
    std::cout << "  wrote " << argv[3] << std::endl;
    return 0;
}
