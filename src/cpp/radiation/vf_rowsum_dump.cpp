/**
 * @file vf_rowsum_dump.cpp
 * @brief Dump per-face view-factor row sums Sum_j F_ij (= terrain view factor,
 *        1 - F_sky) from a saved production H-matrix (.hmat) by computing F*1.
 *
 * This is the model's actual per-face floor->terrain coupling, to compare
 * against the geometric DEM horizon-scan ground truth (haworth_svf_check.py)
 * and so test whether the PSR-floor radiation is under-counted by VF truncation.
 *
 * Usage: vf_rowsum_dump <hmatrix.hmat> <out_rowsum.npy>
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <algorithm>
#include <cmath>

#include "hmatrix_view_factor.h"

namespace {
template <typename T>
void write_npy(const std::string& path, const std::vector<T>& data,
               const std::vector<size_t>& shape, const char* dtype) {
    std::ofstream f(path, std::ios::binary);
    std::string sh = "(";
    for (size_t i = 0; i < shape.size(); ++i)
        sh += std::to_string(shape[i]) + (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : ""));
    sh += ")";
    std::string hdr = "{'descr': '" + std::string(dtype) +
                      "', 'fortran_order': False, 'shape': " + sh + ", }";
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
            static_cast<std::streamsize>(data.size() * sizeof(T)));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: vf_rowsum_dump <hmatrix.hmat> <out_rowsum.npy>\n";
        return 2;
    }
    std::string hmat_path = argv[1], out_path = argv[2];

    radiation::HMatrixViewFactor H;
    std::cout << "Loading H-matrix (mmap): " << hmat_path << std::endl;
    if (!H.readBinaryMmap(hmat_path)) {
        std::cout << "  mmap load failed; trying in-RAM readBinary..." << std::endl;
        if (!H.readBinary(hmat_path)) {
            std::cerr << "  ERROR: could not load H-matrix" << std::endl;
            return 1;
        }
    }
    size_t N = H.numRows();
    std::cout << "  N = " << N << " faces" << std::endl;

    std::vector<double> ones(N, 1.0), rowsum(N, 0.0);
    H.matvec(ones, rowsum);          // rowsum[i] = Sum_j F_ij = terrain VF

    // stats
    double mn = 1e30, mx = -1e30, sum = 0.0;
    for (double r : rowsum) { mn = std::min(mn, r); mx = std::max(mx, r); sum += r; }
    std::cout << "  Sum_j F_ij (terrain VF, all faces): min=" << mn
              << " mean=" << (sum / std::max<size_t>(1, N)) << " max=" << mx << std::endl;
    std::cout << "  -> F_sky range = [" << (1.0 - mx) << ", " << (1.0 - mn) << "]" << std::endl;

    write_npy(out_path, rowsum, {N}, "<f8");
    std::cout << "  wrote " << out_path << std::endl;
    return 0;
}
