"""
Global kinetic fit of BLI sensorgrams: k_on, k_off and K_D = k_off / k_on.

1:1 (Langmuir) model (ligand L on the tip, analyte A in solution at constant concentration C):
    A + L <-> AL,     dR/dt = k_on * C * (R_max - R) - k_off * R

  association  (0 <= t <= t_d):  R(t) = R_eq * (1 - exp(-k_obs * t))
                                 k_obs = k_on * C + k_off,   R_eq = R_max * C / (C + K_D)
  dissociation (t > t_d, C = 0): R(t) = R(t_d) * exp(-k_off * (t - t_d))

Heterogeneous-ligand model (--model 2site): the sum of TWO independent 1:1 populations,
each with its own k_on, k_off, R_max. Use it when the residuals of the 1:1 fit show a fast
and a slow phase. Be careful with the interpretation: the second "site" may be a real second
binding mode, but just as well ligand heterogeneity, avidity, non-specific binding or drift.
The model fits better by construction (more parameters) -- that alone proves nothing.

All concentrations of one analyte are fitted SIMULTANEOUSLY with shared rate constants
("global fit"). R_max is shared too (--rmax global, default) or fitted per sensor
(--rmax local, use when ligand loading differs between tips).

Units: t [s], R [nm], C [M], k_on [1/(M*s)], k_off [1/s], K_D [M].

As an independent cross-check, every association phase is also fitted on its own with a
single exponential; for a true 1:1 interaction k_obs is linear in C, so a straight line of
k_obs vs C gives k_on (slope) and k_off (intercept). If that plot is not a rising straight
line, the 1:1 assumption is violated.

Usage:
    python fit_kinetics.py
    python fit_kinetics.py --model 2site
    python fit_kinetics.py "path/to/run folder" --rmax local --reference double
    python fit_kinetics.py --exclude-conc 10000    # drop concentrations [nM] from the fit
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import sensorgram_tools as st

MIN_RESOLVABLE_DECAY = 0.05  # [-] k_off * t_dissociation below which k_off is reported as unresolved (< 5 % decay)


# ============================================
# MODELS
# ============================================
def langmuir_1to1(time_s, conc_M, t_dissoc_s, k_on, k_off, R_max):
    """
    Analytical 1:1 sensorgram (association followed by dissociation into buffer).
    time_s [s] (array, 0 = start of association), conc_M [M], t_dissoc_s [s],
    k_on [1/(M*s)], k_off [1/s], R_max [nm]  ->  R [nm]
    """
    k_obs   = k_on * conc_M + k_off                     # [1/s]
    R_eq    = R_max * k_on * conc_M / k_obs             # [nm] = R_max * C / (C + K_D)
    t_assoc = np.minimum(time_s, t_dissoc_s)            # [s] time spent in association
    R_assoc = R_eq * (1.0 - np.exp(-k_obs * t_assoc))   # [nm]
    t_dis   = np.maximum(time_s - t_dissoc_s, 0.0)      # [s] time spent in dissociation
    return R_assoc * np.exp(-k_off * t_dis)             # [nm] equals R_assoc during association


def multi_site(time_s, conc_M, t_dissoc_s, k_on, k_off, R_max):
    """Sum of independent 1:1 sites. k_on, k_off, R_max: sequences with one entry per site."""
    return sum(langmuir_1to1(time_s, conc_M, t_dissoc_s, k_on[j], k_off[j], R_max[j]) for j in range(len(k_on)))


# ============================================
# GLOBAL FIT
# ============================================
class GlobalFit:
    """
    Global least-squares fit of `n_sites` independent 1:1 sites to all curves of one analyte.

    Parameter vector p = [log10 k_on_1, log10 k_off_1, ..., log10 k_on_n, log10 k_off_n, R_max block]
    with the R_max block of shape (n_sites, n_rmax), n_rmax = 1 (global) or n_curves (local).
    Rate constants are fitted on log scale: positive by construction and well conditioned
    although k_on (~1e3..1e7) and k_off (~1e-4..1) differ by many decades.
    """

    def __init__(self, curves, t_dissoc_s, n_sites, rmax_mode):
        self.curves     = curves
        self.t_dissoc_s = t_dissoc_s
        self.n_sites    = n_sites
        self.n_rmax     = 1 if rmax_mode == "global" else len(curves)

    def unpack(self, p):
        """p -> k_on [1/(M*s)] (n_sites,), k_off [1/s] (n_sites,), R_max [nm] (n_sites, n_rmax)."""
        rates = 10.0 ** p[:2 * self.n_sites].reshape(self.n_sites, 2)
        R_max = p[2 * self.n_sites:].reshape(self.n_sites, self.n_rmax)
        return rates[:, 0], rates[:, 1], R_max

    def predict(self, p, curve_index):
        k_on, k_off, R_max = self.unpack(p)
        c = self.curves[curve_index]
        return multi_site(c["time_s"], c["conc_M"], self.t_dissoc_s, k_on, k_off,
                          R_max[:, 0 if self.n_rmax == 1 else curve_index])

    def residuals(self, p):
        return np.concatenate([self.predict(p, i) - c["response_nm"] for i, c in enumerate(self.curves)])

    def fit(self, k_on_start, k_off_start, R_max_start):
        """Start values: one entry per site [1/(M*s)], [1/s], [nm]. Returns scipy result + covariance."""
        p0 = np.r_[np.column_stack([np.log10(k_on_start), np.log10(k_off_start)]).ravel(),
                   np.repeat(np.asarray(R_max_start, float), self.n_rmax)]
        lower = np.r_[np.full(2 * self.n_sites, -np.inf), np.zeros(self.n_sites * self.n_rmax)]  # R_max >= 0
        self.result = least_squares(self.residuals, p0, bounds=(lower, np.inf),
                                    method="trf", x_scale="jac")  # bounded trust-region least squares

        # Parameter covariance from the Jacobian: cov = s^2 * (J^T J)^-1.
        # CAUTION: assumes independent, identically distributed residuals. Sensorgram residuals are
        # strongly autocorrelated (5 Hz sampling, systematic misfit), so these standard errors are
        # optimistic, often by an order of magnitude. Treat them as a lower bound on the uncertainty.
        n_points, n_params = self.result.fun.size, self.result.x.size
        s2 = np.sum(self.result.fun ** 2) / (n_points - n_params)  # [nm^2] residual variance
        self.cov = s2 * np.linalg.pinv(self.result.jac.T @ self.result.jac)
        return self.result

    def summary(self):
        """One dict per site with k_on, k_off, K_D and 95 % intervals. Sites sorted fast -> slow k_off."""
        p, cov = self.result.x, self.cov
        rows = []
        for j in range(self.n_sites):
            i_on, i_off = 2 * j, 2 * j + 1
            se_on  = np.sqrt(cov[i_on, i_on])
            se_off = np.sqrt(cov[i_off, i_off])
            log_KD = p[i_off] - p[i_on]
            se_KD  = np.sqrt(cov[i_on, i_on] + cov[i_off, i_off] - 2.0 * cov[i_on, i_off])  # k_on/k_off correlation matters
            _, _, R_max = self.unpack(p)
            rows.append({
                "site":              j + 1,
                "k_on_per_M_s":      10.0 ** p[i_on],
                "k_on_CI95_low":     10.0 ** (p[i_on] - 1.96 * se_on),
                "k_on_CI95_high":    10.0 ** (p[i_on] + 1.96 * se_on),
                "k_off_per_s":       10.0 ** p[i_off],
                "k_off_CI95_low":    10.0 ** (p[i_off] - 1.96 * se_off),
                "k_off_CI95_high":   10.0 ** (p[i_off] + 1.96 * se_off),
                "K_D_M":             10.0 ** log_KD,
                "K_D_CI95_low":      10.0 ** (log_KD - 1.96 * se_KD),
                "K_D_CI95_high":     10.0 ** (log_KD + 1.96 * se_KD),
                "R_max_nm":          R_max[j, 0] if self.n_rmax == 1 else np.nan,
                "corr_log_kon_koff": cov[i_on, i_off] / (se_on * se_off),
            })
        rows.sort(key=lambda r: -r["k_off_per_s"])  # site labels are arbitrary; fix order: fast first
        for j, r in enumerate(rows):
            r["site"] = j + 1
        return rows

    def rmse_nm(self):
        return np.sqrt(np.mean(self.result.fun ** 2))


# ============================================
# CROSS-CHECK: k_obs vs C
# ============================================
def kobs_linear_check(curves, t_dissoc_s):
    """Per-curve exponential fit of the association, then k_obs = k_on * C + k_off by linear regression."""
    rows = []
    for c in curves:
        assoc = c["time_s"] <= t_dissoc_s
        rows.append({"conc_M": c["conc_M"], **st.fit_association_exponential(c["time_s"][assoc], c["response_nm"][assoc])})
    per_curve = pd.DataFrame(rows)
    slope, intercept = np.polyfit(per_curve.conc_M, per_curve.k_obs_per_s, 1)  # [1/(M*s)], [1/s]
    return per_curve, slope, intercept


# ============================================
# PLOTTING
# ============================================
def plot_fits(fit_outputs, t_dissoc_s, model_name):
    """Rows = analytes; columns = data + global fit, residuals, k_obs vs C cross-check."""
    n = len(fit_outputs)
    fig, axes = plt.subplots(n, 3, figsize=(17, 4.6 * n), constrained_layout=True, squeeze=False,
                             gridspec_kw={"width_ratios": [2.2, 1.6, 1.2]})
    colors = st.concentration_colors([c["conc_nM"] for f in fit_outputs for c in f["curves"]])

    for row, f in enumerate(fit_outputs):
        ax_fit, ax_res, ax_kobs = axes[row]
        model = f["model"]
        for i, c in enumerate(f["curves"]):
            prediction = model.predict(model.result.x, i)
            color = colors[c["conc_nM"]]
            ax_fit.plot(c["time_s"], c["response_nm"], color=color, lw=1.0, alpha=0.55, label=f"{c['conc_nM']:g} nM")
            ax_fit.plot(c["time_s"], prediction, color="k", lw=0.9, ls="--")
            ax_res.plot(c["time_s"], c["response_nm"] - prediction, color=color, lw=0.8)
        ax_fit.plot([], [], color="k", ls="--", label=f"{model_name} global fit")

        for ax in (ax_fit, ax_res):
            ax.axvline(t_dissoc_s, color="0.4", ls=":", lw=0.8)
            ax.axhline(0, color="k", lw=0.5)
            ax.grid(alpha=0.3)
            ax.set_xlabel("Time since association start [s]")
        title = "\n".join(f"site {s['site']}: k_on={s['k_on_per_M_s']:.2g} 1/(M s), k_off={s['k_off_per_s']:.2g} 1/s, "
                               f"K_D={s['K_D_M'] * 1e6:.3g} µM" for s in f["sites"])
        ax_fit.set_title(f"{f['analyte']}\n{title}", fontsize=9)
        ax_fit.set_ylabel("Response [nm]")
        ax_fit.legend(fontsize=7, ncol=2)
        ax_res.set_ylabel("Residual (data - model) [nm]")
        ax_res.set_title(f"Residuals, RMSE = {model.rmse_nm() * 1e3:.1f} pm\n(structure = model misspecification)",
                         fontsize=9)

        pc = f["kobs_per_curve"]
        conc_grid_M = np.linspace(0, pc.conc_M.max(), 50)
        ax_kobs.plot(pc.conc_M * 1e6, pc.k_obs_per_s, "o", color="C0", label="per-curve exponential fit")
        ax_kobs.plot(conc_grid_M * 1e6, f["kobs_slope"] * conc_grid_M + f["kobs_intercept"], "C0-",
                     label=f"linear: k_on={f['kobs_slope']:.2g}, k_off={f['kobs_intercept']:.2g}")
        for s in f["sites"]:
            ax_kobs.plot(conc_grid_M * 1e6, s["k_on_per_M_s"] * conc_grid_M + s["k_off_per_s"], "k--",
                         lw=1.2 if s["site"] == 1 else 0.7, label=f"global fit, site {s['site']}")
        ax_kobs.set_xlabel("Analyte concentration [µM]")
        ax_kobs.set_ylabel("k_obs [1/s]")
        ax_kobs.set_title("Cross-check: k_obs = k_on C + k_off", fontsize=9)
        ax_kobs.grid(alpha=0.3)
        ax_kobs.legend(fontsize=7)

    fig.suptitle(f"Global {model_name} kinetic fit")
    return fig


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_folder", nargs="?", default=st.DEFAULT_DATA_FOLDER, type=Path)
    parser.add_argument("--model", default="1to1", choices=["1to1", "2site"],
                        help="1:1 Langmuir (default) or heterogeneous ligand with two independent sites")
    parser.add_argument("--reference", default="double", choices=["none", "reference", "zero", "double"])
    parser.add_argument("--rmax", default="global", choices=["global", "local"],
                        help="share R_max across concentrations (global) or fit one per sensor (local)")
    parser.add_argument("--exclude-conc", type=float, nargs="*", default=[],
                        help="analyte concentrations [nM] to leave out of the fit")
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    output_folder = Path(__file__).parent / "results"
    output_folder.mkdir(exist_ok=True)

    data       = st.load_run(args.data_folder)
    referenced = st.reference_subtract(st.align_kinetic_steps(data), mode=args.reference)
    referenced = referenced[(referenced.analyte_conc_nM > 0) & ~referenced.analyte_conc_nM.isin(args.exclude_conc)]
    t_dissoc_s = st.dissociation_start_s(referenced)  # [s]

    fit_outputs, table_rows = [], []
    for analyte, d in referenced.groupby("analyte"):
        curves = [{
            "conc_nM":     conc,
            "conc_M":      conc * st.NM_TO_M,            # [M]
            "time_s":      s.time_aligned_s.to_numpy(),  # [s]
            "response_nm": s.response_nm.to_numpy(),     # [nm]
        } for conc, s in d.groupby("analyte_conc_nM")]

        # Cross-check, also used for start values (with physically plausible fallbacks)
        kobs_per_curve, slope, intercept = kobs_linear_check(curves, t_dissoc_s)
        R_peak_nm = max(np.max(c["response_nm"]) for c in curves)  # [nm]

        # Always fit 1:1 first; it seeds the 2-site fit (site 1 faster, site 2 slower than the 1:1 average)
        model = GlobalFit(curves, t_dissoc_s, n_sites=1, rmax_mode=args.rmax)
        model.fit([slope if slope > 0 else 1e5], [intercept if intercept > 0 else 1e-2], [1.5 * R_peak_nm])
        if args.model == "2site":
            k_on_1to1, k_off_1to1, _ = model.unpack(model.result.x)
            model = GlobalFit(curves, t_dissoc_s, n_sites=2, rmax_mode=args.rmax)
            model.fit([3.0 * k_on_1to1[0], 0.1 * k_on_1to1[0]],
                      [3.0 * k_off_1to1[0], 0.1 * k_off_1to1[0]],
                      [R_peak_nm, 0.5 * R_peak_nm])
        if not model.result.success:
            print(f"WARNING [{analyte}]: optimiser did not converge: {model.result.message}")

        sites = model.summary()
        # A k_off is only resolved if the dissociation window shows measurable decay. Below that the
        # optimiser can push k_off towards 0 freely and the Jacobian CI collapses (falsely narrow).
        dissoc_duration_s = max(c["time_s"].max() for c in curves) - t_dissoc_s       # [s]
        k_off_limit_per_s = MIN_RESOLVABLE_DECAY / dissoc_duration_s                   # [1/s]
        for s in sites:
            s["k_off_resolved"] = s["k_off_per_s"] >= k_off_limit_per_s
        for s in sites:
            table_rows.append({"analyte": analyte, "model": args.model, **s, "rmse_nm": model.rmse_nm(),
                               "kobs_lin_k_on_per_M_s": slope, "kobs_lin_k_off_per_s": intercept,
                               "reference_mode": args.reference, "rmax_mode": args.rmax,
                               "concs_nM": " ".join(f"{c['conc_nM']:g}" for c in curves)})
        fit_outputs.append({"analyte": analyte, "curves": curves, "model": model, "sites": sites,
                            "kobs_per_curve": kobs_per_curve, "kobs_slope": slope, "kobs_intercept": intercept})

        # Console report (ASCII 'uM' on purpose: Windows consoles often cannot print 'µ')
        print(f"\n{analyte}  ({args.model}, R_max {args.rmax}, {args.reference} referencing)"
              f"  RMSE = {model.rmse_nm() * 1e3:.1f} pm")
        for s in sites:
            print(f"  site {s['site']}: k_on  = {s['k_on_per_M_s']:.3g} 1/(M s)  "
                  f"[{s['k_on_CI95_low']:.3g}, {s['k_on_CI95_high']:.3g}]")
            print(f"          k_off = {s['k_off_per_s']:.3g} 1/s  "
                  f"[{s['k_off_CI95_low']:.3g}, {s['k_off_CI95_high']:.3g}]")
            print(f"          K_D   = {s['K_D_M'] * 1e6:.3g} uM  "
                  f"[{s['K_D_CI95_low'] * 1e6:.3g}, {s['K_D_CI95_high'] * 1e6:.3g}]   R_max = {s['R_max_nm']:.3g} nm")
            if not s["k_off_resolved"]:
                print(f"          !! k_off below the resolution of the {dissoc_duration_s:.0f} s dissociation "
                      f"(< {k_off_limit_per_s:.1e} 1/s): k_off, K_D and their CIs are NOT identifiable.")
        print(f"  k_obs-vs-C cross-check: k_on = {slope:.3g} 1/(M s), k_off = {intercept:.3g} 1/s")
        if slope <= 0 or intercept <= 0:
            print("  -> k_obs does not rise linearly with C: a single exponential does not describe the"
                  " association, i.e. the data are not 1:1.")
    print("\n95 % CI are from the Jacobian and assume uncorrelated residuals -> optimistic lower bounds.")

    tag = f"{args.model}_{args.rmax}"
    pd.DataFrame(table_rows).to_csv(output_folder / f"kinetic_fit_{tag}.csv", index=False)
    plot_fits(fit_outputs, t_dissoc_s, "1:1" if args.model == "1to1" else "2-site").savefig(
        output_folder / f"kinetic_fit_{tag}.png", dpi=150)
    print(f"Results written to {output_folder}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
