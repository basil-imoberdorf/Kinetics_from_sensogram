"""
Load and plot Octet BLI sensorgrams from a folder of .frd files.

Produces two figures:
  1. raw_overview      : full raw traces of every sensor (all assay steps shaded),
                         split into ligand-loaded and reference (bare SA) sensors.
  2. referenced_traces : association + dissociation of the ligand-loaded sensors,
                         aligned at t = 0 and referenced (default: double referencing).

Usage:
    python plot_sensorgrams.py                                  # default data folder
    python plot_sensorgrams.py "path/to/run folder"
    python plot_sensorgrams.py --reference none --export-csv    # unreferenced + tidy CSV
    python plot_sensorgrams.py --no-show                        # only save PNGs
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import sensorgram_tools as st


# ============================================
# PLOTS
# ============================================
def plot_raw_overview(data):
    """One panel per analyte x (reference / ligand-loaded) sensor group: full raw traces with step shading."""
    analytes = sorted(data.analyte.unique())
    fig, axes = plt.subplots(2, len(analytes), figsize=(6.5 * len(analytes), 8), sharex=True,
                             constrained_layout=True, squeeze=False)
    colors = st.concentration_colors(data["analyte_conc_nM"].unique())

    for col, analyte in enumerate(analytes):
        for row, loaded in enumerate([False, True]):
            ax = axes[row, col]
            d = data[(data.analyte == analyte) & (data.ligand_loaded == loaded)]
            # Shade steps using the first sensor of the group (all sensors of one column share the timing)
            first = d[d.sensor == d.sensor.iloc[0]]
            for _, step in first.groupby("step_index"):
                ax.axvspan(step.time_s.min(), step.time_s.max(),
                           color=st.STEP_COLORS.get(step.step_type.iloc[0], "w"), lw=0)
                ax.text(step.time_s.mean(), 1.0, step.step_type.iloc[0], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=7)
            for sensor, s in sorted(d.groupby("sensor"), key=lambda g: -g[1].analyte_conc_nM.iloc[0]):
                conc = s.analyte_conc_nM.iloc[0]
                ax.plot(s.time_s, s.response_nm, lw=1, color=colors[conc], label=f"{sensor}: {conc:g} nM")
            ax.set_title(f"{analyte} | {'ligand loaded' if loaded else 'reference (no ligand)'}", pad=14)
            ax.set_ylabel("Response [nm]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Raw sensorgrams (all steps)")
    return fig


def plot_referenced_traces(referenced, mode):
    """Association/dissociation of the ligand-loaded sensors after referencing, aligned at t = 0."""
    analytes = sorted(referenced.analyte.unique())
    fig, axes = plt.subplots(1, len(analytes), figsize=(6.5 * len(analytes), 4.5), sharey=True,
                             constrained_layout=True, squeeze=False)
    colors = st.concentration_colors(referenced["analyte_conc_nM"].unique())
    t_dissoc_s = st.dissociation_start_s(referenced)  # [s]

    for ax, analyte in zip(axes.flat, analytes):
        d = referenced[referenced.analyte == analyte]
        for conc, s in sorted(d.groupby("analyte_conc_nM"), key=lambda g: -g[0]):
            ax.plot(s.time_aligned_s, s.response_nm, lw=1.2, color=colors[conc], label=f"{conc:g} nM")
        ax.axvline(t_dissoc_s, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{analyte} ({mode} referencing)")
        ax.set_xlabel("Time since association start [s]")
        ax.grid(alpha=0.3)
        ax.legend(title="Analyte", fontsize=8)
    axes.flat[0].set_ylabel("Response [nm]")
    fig.suptitle("Association / dissociation on ligand-loaded sensors")
    return fig


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_folder", nargs="?", default=st.DEFAULT_DATA_FOLDER, type=Path)
    parser.add_argument("--reference", default="double", choices=["none", "reference", "zero", "double"],
                        help="referencing mode for the kinetic plot (default: double)")
    parser.add_argument("--export-csv", action="store_true", help="write the tidy long table to results/")
    parser.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use("Agg")

    output_folder = Path(__file__).parent / "results"
    output_folder.mkdir(exist_ok=True)

    data = st.load_run(args.data_folder)
    print(st.sensor_table(data).to_string(index=False))

    aligned    = st.align_kinetic_steps(data)
    referenced = st.reference_subtract(aligned, mode=args.reference)

    plot_raw_overview(data).savefig(output_folder / "raw_overview.png", dpi=150)
    plot_referenced_traces(referenced, args.reference).savefig(output_folder / "referenced_traces.png", dpi=150)

    if args.export_csv:
        data.to_csv(output_folder / "sensorgrams_long.csv", index=False)  # e.g. for MATLAB readtable
        print(f"Tidy data written to {output_folder / 'sensorgrams_long.csv'}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
