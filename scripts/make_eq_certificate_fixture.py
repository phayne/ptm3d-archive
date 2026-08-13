#!/usr/bin/env python3
"""Write a two-population 'PTMT' fixture for the retrospective equilibration
certificate, so the reduction can be checked END TO END through
`build/bin/ptm_eq_certify` and not only inside `test_eq_certify`.

THE TWO POPULATIONS
  SETTLED    the zero-forcing conductive steady profile, the only exactly-known
             state in the system: eps*sigma*T0^4 = Qb, then
             T[j+1] = T[j] + Qb*dz[j]/k(T[j], z[j]), k at the SHALLOWER node
             (the engine's own interior stencil). Its interface flux is Qb at
             every depth, so the certificate must PASS it.
  UNSETTLED  the same column started isothermal at T0. No interior gradient has
             developed, so q = 0 and r = -Qb exactly, and the certificate must
             FAIL it -- at |r|/(4 eps sigma T0^3), which for the lunar numbers
             is 6.0 K against a 0.1 K target.
  ARRESTED   optional third population: the UNSETTLED column, rounded to
             float32 and written unchanged. It is bit-frozen from one pass to
             the next, so every increment-based convergence test calls it
             converged. The certificate must still fail it.

THE GRID IS NOT RECONSTRUCTED HERE. z and kc come from `dump_thermal_grid`,
which prints the arrays `thermal::Profile` actually builds. A transcription of
the grid that is wrong by a few percent is indistinguishable from an
energy-conservation defect, and that ambiguity is what blocked the 2026-08-08
flux audit; this script refuses to guess.

USAGE
  build/bin/dump_thermal_grid Moon 10 15 32 30.0 > /tmp/grid.csv
  python3 scripts/make_eq_certificate_fixture.py --grid-csv /tmp/grid.csv \\
      --out-dir <dir> [--faces-per-population 64] [--Qb 0.018] [--eps 0.95]
      [--chi 2.7]

Writes <dir>/state.bin plus <dir>/statistics/{temperature_max,temperature_min,
direct_solar_max}.bin, which is exactly the input pair ptm_eq_certify takes:

  build/bin/ptm_eq_certify --state <dir>/state.bin --stats <dir>/statistics \\
      --body Moon --m 10 --n 15 --b 32 --period-ref annual \\
      --window-seconds 1e12 --csv <dir>/perface.csv

Exit status 0. This script only writes a fixture; it asserts nothing.
"""
import argparse
import csv
import os
import struct
import sys

PTMT_MAGIC = 0x50544D54


def read_grid_csv(path):
    """z[m] and kc[W/m/K] as `thermal::Profile` built them."""
    z, kc = [], []
    with open(path, newline="") as fh:
        rows = [r for r in fh if not r.startswith("#")]
    for row in csv.DictReader(rows):
        z.append(float(row["z_m"]))
        kc.append(float(row["kc"]))
    if len(z) < 4:
        sys.exit(f"REFUSED: '{path}' holds only {len(z)} layers; that is not a "
                 "column. Regenerate it with build/bin/dump_thermal_grid.")
    return z, kc


def steady_profile(z, kc, Qb, eps, chi, sigma):
    """The closed-form zero-forcing steady state. No iteration, no solver."""
    r350 = chi / 350.0 ** 3
    T = [(Qb / (eps * sigma)) ** 0.25]
    for j in range(1, len(z)):
        k = kc[j - 1] * (1.0 + r350 * T[j - 1] ** 3)
        T.append(T[j - 1] + Qb * (z[j] - z[j - 1]) / k)
    # Bottom node on the engine's own bottom BC, which uses k at the DEEPER
    # node -- and k there depends on that node's temperature, so iterate to the
    # fixed point the solver reaches rather than taking one pass.
    for _ in range(4):
        k_last = kc[-1] * (1.0 + r350 * T[-1] ** 3)
        T[-1] = T[-2] + Qb * (z[-1] - z[-2]) / k_last
    return T


def write_ptmt(path, columns):
    """columns: list of per-face T lists, all the same length. Layer-major."""
    n_faces = len(columns)
    n_layers = len(columns[0])
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", PTMT_MAGIC, n_faces, n_layers))
        for l in range(n_layers):
            fh.write(struct.pack(f"<{n_faces}f", *[c[l] for c in columns]))


def write_stats_bin(path, values):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", len(values)))
        fh.write(struct.pack(f"<{len(values)}f", *values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-csv", required=True,
                    help="output of build/bin/dump_thermal_grid")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--faces-per-population", type=int, default=64)
    ap.add_argument("--Qb", type=float, default=0.018)
    ap.add_argument("--eps", type=float, default=0.95)
    ap.add_argument("--chi", type=float, default=2.7)
    # The engine's sigma, not CODATA's. They differ in the 6th digit and the
    # difference is visible in T0 (24.042613 vs 24.042758 K).
    ap.add_argument("--sigma", type=float, default=5.67051196e-8)
    ap.add_argument("--no-arrested", action="store_true")
    a = ap.parse_args()

    z, kc = read_grid_csv(a.grid_csv)
    settled = steady_profile(z, kc, a.Qb, a.eps, a.chi, a.sigma)
    T0 = settled[0]
    unsettled = [T0] * len(z)
    # The bottom node still obeys the bottom BC, so Qb stays recoverable from
    # the state -- which is how the certificate reads it.
    r350 = a.chi / 350.0 ** 3
    for _ in range(4):
        k_last = kc[-1] * (1.0 + r350 * unsettled[-1] ** 3)
        unsettled[-1] = unsettled[-2] + a.Qb * (z[-1] - z[-2]) / k_last

    N = a.faces_per_population
    columns, labels = [], []
    for name, col in (("settled", settled), ("unsettled", unsettled)):
        columns += [list(col)] * N
        labels += [name] * N
    if not a.no_arrested:
        # Bit-identical to `unsettled` after the float32 write. Present so the
        # fixture carries the arrest case explicitly, and so the per-face CSV
        # shows it landing on the same verdict as the state it is a copy of.
        columns += [list(unsettled)] * N
        labels += ["arrested"] * N

    os.makedirs(os.path.join(a.out_dir, "statistics"), exist_ok=True)
    state = os.path.join(a.out_dir, "state.bin")
    write_ptmt(state, columns)

    # Zero swing on every face: the surface never moves, so the snapshot IS the
    # cycle mean and the certificate's verdicts are definitive rather than
    # INDETERMINATE. That is true by construction here -- there is no forcing.
    n = len(columns)
    surf = [c[0] for c in columns]
    stats = os.path.join(a.out_dir, "statistics")
    write_stats_bin(os.path.join(stats, "temperature_max.bin"), surf)
    write_stats_bin(os.path.join(stats, "temperature_min.bin"), surf)
    write_stats_bin(os.path.join(stats, "direct_solar_max.bin"), [0.0] * n)

    with open(os.path.join(a.out_dir, "labels.csv"), "w") as fh:
        fh.write("face,population\n")
        for i, lab in enumerate(labels):
            fh.write(f"{i},{lab}\n")

    print(f"wrote {state}  ({n} faces x {len(z)} layers)")
    print(f"  T0 = {T0:.6f} K   z_max = {z[-1]:.6f} m   Qb = {a.Qb} W/m^2")
    print(f"  populations: {' '.join(sorted(set(labels)))}, {N} faces each")
    print(f"  statistics + labels.csv alongside")
    return 0


if __name__ == "__main__":
    sys.exit(main())
