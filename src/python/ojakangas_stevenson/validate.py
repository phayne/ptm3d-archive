"""Validate the O&S implementation against the published Table I.

Usage:
    python -m ojakangas_stevenson.validate
    python -m ojakangas_stevenson.validate --plots --save-dir ./validation_plots
    python -m ojakangas_stevenson.validate --method exact     # informational only

The gate (exit 0) runs in PAPER mode (constants.PAPER, C_R = 0.5, Eq. 2.8)
and requires EVERY Table I entry -- average thickness, 4 t22 and t20 + 2 t22
for all four cases -- to agree with the paper within 5% OR 0.1 km.

History: before 2026-10 the thickness solver was not Eq. 2.8 (it dropped the
factor 2 on D and added H linearly). That made C_R = 0.5 give shells ~45% too
thick and led to the accidental fit C_R = 0.85, with a 30% tolerance on the
average only. With Eq. 2.8 the paper's own C_R = 0.5 reproduces all twelve
entries (physics review, 2026-10-03).
"""

import argparse
import os
import sys

import numpy as np

from . import constants as C
from .surface_temperature import OSSurfaceTemperature
from .shell_thickness import IceShellModel, table1_case, TABLE1_CASES

# Paper reference values (thermal paper Table I, p. 230)
PAPER_VALUES = {
    'maxwell_h0': ('Maxwell, H=0', {'avg_km': 23.0, '4t22_km': 6.6, 't20_2t22_km': -0.1}),
    'maxwell_h10': ('Maxwell, H=10', {'avg_km': 21.0, '4t22_km': 4.6, 't20_2t22_km': 0.4}),
    'flow_ng3': ('Flow law n_G=3', {'avg_km': 25.0, '4t22_km': 4.4, 't20_2t22_km': 1.1}),
    'flow_ng52': ('Flow law n_G=5.2', {'avg_km': 13.0, '4t22_km': 2.0, 't20_2t22_km': 0.8}),
}
REL_TOL = 0.05     # 5%
ABS_TOL_KM = 0.1   # or 0.1 km, whichever is looser


def entry_passes(computed, paper):
    """True if |computed - paper| <= max(5% |paper|, 0.1 km)."""
    return abs(computed - paper) <= max(REL_TOL * abs(paper), ABS_TOL_KM)


def validate_surface_temperatures():
    """Check the Appendix A surface temperatures (Eq. A.9 / A.15)."""
    print("=" * 60)
    print("Surface Temperature (Appendix A, Eqs. A.9, A.15-A.18)")
    print("=" * 60)
    model = OSSurfaceTemperature()
    T_eq = float(model.temperature(np.pi / 2))
    T_pole = float(model.temperature(0.0))
    T_avg = model.global_mean_temperature()
    T_pole_expected = (((1 - 0.5) * C.SOLAR_FLUX_JUPITER * C.OBLIQUITY
                        / (np.sqrt(2.0) * np.pi)) / C.STEFAN_BOLTZMANN) ** 0.25
    print(f"  Equator: {T_eq:.1f} K  (paper text: ~110 K)")
    print(f"  Pole:    {T_pole:.1f} K  (Eq. A.18 at phi = 0: {T_pole_expected:.1f} K; "
          f"the paper's text says ~52 K, which is S0 i/pi, not A.15)")
    print(f"  Mean:    {T_avg:.1f} K  (paper text: ~100 K)")
    ok = (abs(T_eq - 110) < 2 and abs(T_pole - T_pole_expected) < 1e-9
          and abs(T_avg - 100) < 5)
    print(f"  Status: {'PASS' if ok else 'FAIL'}\n")
    return ok


def build_models(c_r=None, method='eq28'):
    """The four Table I models."""
    temp_model = OSSurfaceTemperature()
    models = {}
    for key in TABLE1_CASES:
        rheology, H = table1_case(key)
        models[key] = IceShellModel(rheology, surface_temp_model=temp_model,
                                    H_flux=H, c_r=c_r, method=method)
    return models


def validate_thickness(models, n_theta=361, n_lam=720):
    """Compare all twelve Table I entries; return (all_ok, results)."""
    print("=" * 60)
    print("Shell Thickness (Table I): pass if within 5% or 0.1 km")
    print("=" * 60)
    header = (f"{'Case':<18} | {'<b>':>6} {'paper':>5} | {'4t22':>6} {'paper':>5} | "
              f"{'t20+2t22':>8} {'paper':>5} | status")
    print(header)
    print("-" * len(header))
    all_ok = True
    results = {}
    for key, model in models.items():
        label, paper = PAPER_VALUES[key]
        theta, lam, b = model.thickness_grid(n_theta, n_lam)
        from .grid import area_mean, harmonics
        h = harmonics(b, theta, lam)
        computed = {'avg_km': area_mean(b, theta, lam) / 1e3,
                    '4t22_km': h['four_t22'] / 1e3,
                    't20_2t22_km': h['t20_2t22'] / 1e3}
        ok = {k: entry_passes(computed[k], paper[k]) for k in paper}
        all_ok &= all(ok.values())
        flags = ''.join('.' if ok[k] else 'X' for k in ('avg_km', '4t22_km', 't20_2t22_km'))
        print(f"{label:<18} | {computed['avg_km']:6.2f} {paper['avg_km']:5.1f} | "
              f"{computed['4t22_km']:6.2f} {paper['4t22_km']:5.1f} | "
              f"{computed['t20_2t22_km']:8.2f} {paper['t20_2t22_km']:5.1f} | "
              f"{'PASS' if all(ok.values()) else 'FAIL ' + flags}")
        results[label] = {
            'computed_avg_km': computed['avg_km'], 'paper_avg_km': paper['avg_km'],
            'computed_4t22_km': computed['4t22_km'], 'paper_4t22_km': paper['4t22_km'],
            'computed_t20_2t22_km': computed['t20_2t22_km'],
            'paper_t20_2t22_km': paper['t20_2t22_km'],
        }
    print(f"\n  Overall: {'PASS' if all_ok else 'FAIL'}\n")
    return all_ok, results


def generate_plots(models, results_dict, save_dir):
    """Generate validation plots."""
    from . import plotting
    os.makedirs(save_dir, exist_ok=True)
    plotting.plot_surface_temperature(
        save_path=os.path.join(save_dir, 'surface_temperature.png'))
    plotting.plot_strain_pattern(save_path=os.path.join(save_dir, 'strain_pattern.png'))
    for key, model in models.items():
        plotting.plot_shell_thickness_contour(
            model, save_path=os.path.join(save_dir, f'thickness_{key}.png'))
        plotting.plot_thickness_vs_colatitude(
            model, save_path=os.path.join(save_dir, f'thickness_profile_{key}.png'))
    plotting.plot_validation_table(
        results_dict, save_path=os.path.join(save_dir, 'validation_table.png'))
    print(f"Plots -> {save_dir}\n")


def run_all_validations(plot=False, save_dir='./validation_plots', c_r=None,
                        method='eq28', parameter_set='paper'):
    """Run the validation. Returns True only for a passing PAPER-mode gate."""
    with C.parameter_set(parameter_set):
        print("\nOjakangas & Stevenson (1989) implementation validation")
        print("=" * 60)
        c_r_val = C.C_R if c_r is None else c_r
        print(C.describe_active().replace(f"C_R={C.C_R}", f"C_R={c_r_val}"))
        print(f"thickness law: {method}\n")
        print(f"  gamma = {C.gamma():.4e}, mu' = {C.mu_prime():.3f}, "
              f"g3/g2 = {C.g3(c_r)/C.g2(c_r):.4f} (11/3), n tau_min = {C.ntau_min():.3f}\n")
        ok_temp = validate_surface_temperatures()
        models = build_models(c_r=c_r, method=method)
        ok_thickness, results = validate_thickness(models)
        if plot:
            generate_plots(models, results, save_dir)
    gate = (parameter_set == 'paper' and method == 'eq28' and c_r in (None, 0.5))
    all_ok = ok_temp and ok_thickness
    if not gate:
        print("NOTE: not the gate configuration (paper set, Eq. 2.8, C_R = 0.5); "
              "results are informational.")
        return False
    print("All validations PASSED." if all_ok else "Some validations FAILED.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Validate the O&S ice shell model against Table I')
    parser.add_argument('--plots', action='store_true', help='Generate validation plots')
    parser.add_argument('--save-dir', default='./validation_plots')
    parser.add_argument('--cr', type=float, default=None,
                        help='Tidal correction factor C_R (default: 0.5, the paper value)')
    parser.add_argument('--method', choices=('eq28', 'exact'), default='eq28',
                        help='thickness law (the gate uses Eq. 2.8)')
    parser.add_argument('--parameter-set', default='paper',
                        choices=sorted(C.PARAMETER_SETS),
                        help='constants (the gate uses the paper set)')
    args = parser.parse_args()
    ok = run_all_validations(plot=args.plots, save_dir=args.save_dir, c_r=args.cr,
                             method=args.method, parameter_set=args.parameter_set)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
