"""
Steady-state (equilibrium) affinity analysis of BLI sensorgrams: plateau response vs concentration.

For a 1:1 interaction at equilibrium:
    R_eq(C) = R_max * C / (K_D + C)          (Langmuir binding isotherm)

K_D is the concentration giving half-maximal response. The analysis uses only the plateau
of each association phase, so it is independent of the rate constants -- a useful
cross-check of the kinetic K_D and the method of choice for fast-on/fast-off binders.

Two ways to obtain R_eq per concentration (--plateau):
  "mean" : mean response over the last --window-s seconds of association (default).
           Only valid if the curve has actually reached equilibrium.
  "fit"  : R_eq from a single-exponential fit of the association (extrapolated plateau).
           Can rescue curves that are close to, but not at, equilibrium -- but extrapolates.

Checks reported:
  - Equilibrium reached?  The slope over the plateau window, relative to the plateau level,
    is reported per curve; a curve still rising there gives a biased (low) R_eq.
  - Is R_max constrained? If the highest concentration is not well above K_D
    (C_max / K_D < ~3), R_max and therefore K_D are extrapolated and strongly correlated.

Units: C [M] (printed in µM), R [nm], K_D [M].

Usage:
    python fit_steady_state.py
    python fit_steady_state.py --plateau fit
    python fit_steady_state.py "path/to/run folder" --window-s 20 --exclude-conc 10000
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import sensorgram_tools as st

MAX_RELATIVE_DRIFT = 0.05  # [-] max. change of R over the plateau window, relative to R_eq, to call it "at equilibrium"
MIN_CMAX_OVER_KD   = 3.0   # [-] below this C_max/K_D ratio R_max is considered poorly constrained
NOISE_FLOOR_SD     = 2.0   # [-] drift smaller than this many noise SDs is not called drift (near-zero signals)
MIN_KOBS_T_ASSOC   = 3.0   # [-] k_obs * t_assoc below this -> < 95 % of the plateau observed, exp. plateau extrapolated


# ============================================
# PLATEAU EXTRACTION
# ============================================
def extract_plateaus(referenced, window_s, method):
    """One row per (analyte, concentration): R_eq [nm] and an equilibrium diagnostic."""
    t_dissoc_s = st.dissociation_start_s(referenced)  # [s]
    rows = []
    for (analyte, conc_nM), s in referenced.groupby(["analyte", "analyte_conc_nM"]):
        assoc  = s[s.step_type == "ASSOC"]
        t      = assoc.time_aligned_s.to_numpy()  # [s]
        y      = assoc.response_nm.to_numpy()     # [nm]
        window = t >= t_dissoc_s - window_s

        slope_nm_per_s = np.polyfit(t[window], y[window], 1)[0]  # [nm/s] residual slope on the plateau
        R_mean_nm      = y[window].mean()                         # [nm]
        exp_fit        = st.fit_association_exponential(t, y)
        R_eq_nm        = R_mean_nm if method == "mean" else exp_fit["R_eq_nm"]

        # Drift = change over the window. It counts only if it is large relative to the plateau AND
        # above the noise; otherwise near-zero signals at low C are flagged for pure noise.
        drift_nm       = abs(slope_nm_per_s * window_s)                  # [nm]
        noise_sd_nm    = y[window].std(ddof=1)                           # [nm]
        relative_drift = drift_nm / max(abs(R_eq_nm), 1e-12)
        drifting       = (relative_drift > MAX_RELATIVE_DRIFT) and (drift_nm > NOISE_FLOOR_SD * noise_sd_nm)

        # Exponential plateau is an extrapolation if less than ~95 % of it was observed (k_obs * t < 3)
        kobs_t_assoc = exp_fit["k_obs_per_s"] * (t[-1] - t[0])          # [-]
        rows.append({
            "analyte":             analyte,
            "conc_nM":             conc_nM,
            "conc_M":              conc_nM * st.NM_TO_M,  # [M]
            "R_eq_nm":             R_eq_nm,
            "R_eq_sd_nm":          noise_sd_nm,  # [nm] noise on the window (for plotting)
            "plateau_slope_nm_s":  slope_nm_per_s,
            "relative_drift":      relative_drift,
            "at_equilibrium":      not drifting,
            "k_obs_per_s":         exp_fit["k_obs_per_s"],
            "kobs_t_assoc":        kobs_t_assoc,
            "use_in_fit":          True if method == "mean" else kobs_t_assoc >= MIN_KOBS_T_ASSOC,
        })
    return pd.DataFrame(rows)


# ============================================
# LANGMUIR ISOTHERM FIT
# ============================================
def langmuir_isotherm(conc_M, R_max, K_D):
    """R_eq [nm] = R_max [nm] * C [M] / (K_D [M] + C [M])"""
    return R_max * conc_M / (K_D + conc_M)


def fit_isotherm(conc_M, R_eq_nm):
    """
    Least-squares fit of the Langmuir isotherm. Parameters p = [R_max [nm], log10 K_D [M]].
    Returns R_max, K_D, 95 % CI of K_D and the R_max-K_D correlation (from the Jacobian;
    with only a handful of points these intervals are approximate).
    """
    def residuals(p):
        return langmuir_isotherm(conc_M, p[0], 10.0 ** p[1]) - R_eq_nm

    p0 = [1.2 * R_eq_nm.max(), np.log10(np.median(conc_M))]  # start: R_max above highest point, K_D mid-range
    result = least_squares(residuals, p0, bounds=([0, -np.inf], [np.inf, np.inf]), method="trf")

    dof = max(conc_M.size - 2, 1)
    s2  = np.sum(result.fun ** 2) / dof  # [nm^2]
    cov = s2 * np.linalg.pinv(result.jac.T @ result.jac)
    se_log_KD = np.sqrt(cov[1, 1])
    return {
        "R_max_nm":        result.x[0],
        "R_max_se_nm":     np.sqrt(cov[0, 0]),
        "K_D_M":           10.0 ** result.x[1],
        "K_D_CI95_low":    10.0 ** (result.x[1] - 1.96 * se_log_KD),
        "K_D_CI95_high":   10.0 ** (result.x[1] + 1.96 * se_log_KD),
        "corr_Rmax_logKD": cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
        "rmse_nm":         np.sqrt(np.mean(result.fun ** 2)),
    }


# ============================================
# PLOTTING
# ============================================
def plot_steady_state(referenced, plateaus, fits, window_s):
    """Row per analyte: association with plateau window marked | isotherm on log-C axis."""
    analytes = sorted(plateaus.analyte.unique())
    fig, axes = plt.subplots(len(analytes), 2, figsize=(13, 4.6 * len(analytes)), constrained_layout=True,
                             squeeze=False, gridspec_kw={"width_ratios": [1.3, 1]})
    colors     = st.concentration_colors(plateaus.conc_nM.unique())
    t_dissoc_s = st.dissociation_start_s(referenced)

    for row, analyte in enumerate(analytes):
        ax_trace, ax_iso = axes[row]
        p, f = plateaus[plateaus.analyte == analyte], fits[analyte]

        d = referenced[(referenced.analyte == analyte) & (referenced.step_type == "ASSOC")]
        for conc, s in d.groupby("analyte_conc_nM"):
            ax_trace.plot(s.time_aligned_s, s.response_nm, color=colors[conc], lw=1, label=f"{conc:g} nM")
        used = p[p.use_in_fit]
        ax_trace.scatter(np.full(len(used), t_dissoc_s - window_s / 2), used.R_eq_nm, color="k", zorder=3, s=14,
                         label="R_eq used")
        ax_trace.axvspan(t_dissoc_s - window_s, t_dissoc_s, color="0.85", lw=0, label="plateau window")
        ax_trace.set_xlabel("Time since association start [s]")
        ax_trace.set_ylabel("Response [nm]")
        ax_trace.set_title(f"{analyte}: association phases", fontsize=10)
        ax_trace.grid(alpha=0.3)
        ax_trace.legend(fontsize=7, ncol=2)

        conc_grid_M = np.logspace(np.log10(p.conc_M.min()) - 0.5, np.log10(max(p.conc_M.max(), f["K_D_M"])) + 1, 200)
        excluded = p[~p.use_in_fit]
        if not excluded.empty:
            ax_iso.plot(excluded.conc_M * 1e6, excluded.R_eq_nm, "x", color="0.4", label="excluded (extrapolated)")
        p = p[p.use_in_fit]
        equilibrium = p.at_equilibrium.to_numpy()
        ax_iso.errorbar(p.conc_M[equilibrium] * 1e6, p.R_eq_nm[equilibrium], yerr=p.R_eq_sd_nm[equilibrium],
                        fmt="o", color="C0", label="R_eq (at equilibrium)")
        if (~equilibrium).any():
            ax_iso.errorbar(p.conc_M[~equilibrium] * 1e6, p.R_eq_nm[~equilibrium], yerr=p.R_eq_sd_nm[~equilibrium],
                            fmt="o", mfc="w", color="C3", label="R_eq (still drifting)")
        ax_iso.plot(conc_grid_M * 1e6, langmuir_isotherm(conc_grid_M, f["R_max_nm"], f["K_D_M"]), "k-",
                    label=f"Langmuir: K_D = {f['K_D_M'] * 1e6:.3g} µM")
        ax_iso.axhline(f["R_max_nm"], color="k", ls=":", lw=0.8, label=f"R_max = {f['R_max_nm']:.3g} nm")
        ax_iso.axvline(f["K_D_M"] * 1e6, color="0.5", ls="--", lw=0.8)
        ax_iso.set_xscale("log")
        y_top = max(p.R_eq_nm.max(), f["R_max_nm"])
        ax_iso.set_ylim(min(0, p.R_eq_nm.min()) - 0.05 * y_top, 1.15 * y_top)  # keep wild excluded points out of view
        ax_iso.set_xlabel("Analyte concentration [µM]")
        ax_iso.set_ylabel("Equilibrium response R_eq [nm]")
        ax_iso.set_title(f"Steady-state isotherm   (C_max / K_D = {p.conc_M.max() / f['K_D_M']:.1f})", fontsize=10)
        ax_trace.set_ylim(ax_iso.get_ylim())
        ax_iso.grid(alpha=0.3, which="both")
        ax_iso.legend(fontsize=7, loc="upper left")

    fig.suptitle("Steady-state affinity analysis")
    return fig


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_folder", nargs="?", default=st.DEFAULT_DATA_FOLDER, type=Path)
    parser.add_argument("--reference", default="double", choices=["none", "reference", "zero", "double"])
    parser.add_argument("--plateau", default="mean", choices=["mean", "fit"],
                        help="R_eq = mean of the last window (mean) or exponential-fit plateau (fit)")
    parser.add_argument("--window-s", type=float, default=30.0, help="plateau window at the end of association [s]")
    parser.add_argument("--exclude-conc", type=float, nargs="*", default=[],
                        help="analyte concentrations [nM] to leave out of the isotherm fit")
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    output_folder = Path(__file__).parent / "results"
    output_folder.mkdir(exist_ok=True)

    data       = st.load_run(args.data_folder)
    referenced = st.reference_subtract(st.align_kinetic_steps(data), mode=args.reference)
    referenced = referenced[(referenced.analyte_conc_nM > 0) & ~referenced.analyte_conc_nM.isin(args.exclude_conc)]

    plateaus = extract_plateaus(referenced, args.window_s, args.plateau)
    fits, rows = {}, []
    for analyte, p_all in plateaus.groupby("analyte"):
        p = p_all[p_all.use_in_fit]
        fits[analyte] = fit_isotherm(p.conc_M.to_numpy(), p.R_eq_nm.to_numpy())  # Langmuir least squares
        f = fits[analyte]
        cmax_over_KD = p.conc_M.max() / f["K_D_M"]

        # Console report (ASCII 'uM' on purpose: Windows consoles often cannot print 'µ')
        print(f"\n{analyte}  ({args.reference} referencing, plateau = {args.plateau}, window = {args.window_s:g} s)")
        print(p_all[["conc_nM", "R_eq_nm", "plateau_slope_nm_s", "relative_drift", "at_equilibrium",
                     "kobs_t_assoc", "use_in_fit"]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        if (~p_all.use_in_fit).any():
            print(f"  !! {(~p_all.use_in_fit).sum()} exponential plateau(s) excluded: k_obs * t_assoc < "
                  f"{MIN_KOBS_T_ASSOC:g}, i.e. the plateau would be an extrapolation.")
        print(f"  K_D   = {f['K_D_M'] * 1e6:.3g} uM  [{f['K_D_CI95_low'] * 1e6:.3g}, {f['K_D_CI95_high'] * 1e6:.3g}]")
        print(f"  R_max = {f['R_max_nm']:.3g} +/- {f['R_max_se_nm']:.2g} nm,  corr(R_max, log K_D) = "
              f"{f['corr_Rmax_logKD']:+.2f},  RMSE = {f['rmse_nm'] * 1e3:.1f} pm")
        print(f"  C_max / K_D = {cmax_over_KD:.1f}")
        if cmax_over_KD < MIN_CMAX_OVER_KD:
            print(f"  !! C_max / K_D < {MIN_CMAX_OVER_KD:g}: the isotherm does not saturate, R_max is extrapolated"
                  " and K_D is correspondingly uncertain. Higher concentrations would be needed.")
        if not p.at_equilibrium.all():
            print(f"  !! {(~p.at_equilibrium).sum()} curve(s) still drift by > {MAX_RELATIVE_DRIFT:.0%} over the"
                  " window: R_eq is not an equilibrium value there (slow phase / non-1:1 behaviour).")
        rows.append({"analyte": analyte, **f, "C_max_over_K_D": cmax_over_KD, "n_points_fitted": len(p),
                     "n_not_at_equilibrium": int((~p.at_equilibrium).sum()),
                     "reference_mode": args.reference, "plateau_method": args.plateau, "window_s": args.window_s})

    plateaus.to_csv(output_folder / "steady_state_plateaus.csv", index=False)
    pd.DataFrame(rows).to_csv(output_folder / "steady_state_fit.csv", index=False)
    plot_steady_state(referenced, plateaus, fits, args.window_s).savefig(output_folder / "steady_state_fit.png", dpi=150)
    print(f"\nResults written to {output_folder}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
