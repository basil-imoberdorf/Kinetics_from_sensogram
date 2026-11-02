"""
Shared helpers for Octet (Sartorius/ForteBio) BLI sensorgram analysis.

Used by plot_sensorgrams.py, fit_kinetics.py and fit_steady_state.py, so that
parsing, alignment and referencing are implemented exactly once.

Data format
-----------
Each .frd file is XML and holds ONE biosensor tip (e.g. "A6"). Inside, every
assay step (Baseline, Loading, Baseline, Association, Dissociation) stores its
time and signal arrays as base64-encoded little-endian float32 vectors
(<AssayXData> = time [s], <AssayYData> = wavelength shift [nm]).

Units
-----
time [s], response [nm] (BLI wavelength shift), concentration [nM] in the tables
(as stored by the instrument). The fits convert concentrations to molar
[M = mol/L], the binding-literature convention for k_on [1/(M*s)] and K_D [M]
(1 M = 1e3 mol/m^3 in strict SI).
"""

import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DATA_FOLDER = Path(__file__).parent / "biotin CK1 short C NtoA muts"

NM_TO_M = 1e-9  # [M/nM] concentration conversion nanomolar -> molar


# ============================================
# PARSING
# ============================================
def decode_float32_array(base64_text):
    """Decode a base64 string into a float64 numpy array (stored as little-endian float32)."""
    return np.frombuffer(base64.b64decode(base64_text), dtype="<f4").astype(float)


def read_frd(file_path):
    """
    Read one Octet .frd file into a tidy (long) DataFrame, one row per data point.

    Columns: file, sensor, sensor_type, step_index, step_type, step_name,
             sample_id, well_type, well, concentration_nM, time_s, response_nm
    """
    root = ET.parse(file_path).getroot()
    experiment_info = root.find("ExperimentInfo")
    sensor_name = experiment_info.findtext("SensorName")  # tip position on the sensor tray, e.g. "A6"
    sensor_type = experiment_info.findtext("SensorType")  # e.g. "SA (Streptavidin)"

    step_tables = []
    for step_index, step in enumerate(root.find("KineticsData").findall("Step")):
        common = step.find("CommonData")
        time_s           = decode_float32_array(step.findtext("AssayXData"))  # [s] cumulative time since sensor start
        response_nm      = decode_float32_array(step.findtext("AssayYData"))  # [nm] interference wavelength shift
        concentration_nM = float(common.findtext("MolarConcentration"))      # [nM] -1 means "not applicable" (buffer)

        step_tables.append(pd.DataFrame({
            "file":             Path(file_path).name,
            "sensor":           sensor_name,
            "sensor_type":      sensor_type,
            "step_index":       step_index,
            "step_type":        step.findtext("StepType"),  # BASELINE / LOADING / ASSOC / DISASSOC
            "step_name":        step.findtext("StepName"),
            "sample_id":        common.findtext("SampleID"),
            "well_type":        common.findtext("WellType"),  # BUFFER / KLOAD / KSAMPLE
            "well":             common.findtext("SampleRow") + common.findtext("SampleLocation"),
            "concentration_nM": np.nan if concentration_nM < 0 else concentration_nM,
            "time_s":           time_s,
            "response_nm":      response_nm,
        }))
    return pd.concat(step_tables, ignore_index=True)


def load_run(data_folder=DEFAULT_DATA_FOLDER):
    """Read every .frd file in a folder and add per-sensor annotations (analyte, ligand loaded?)."""
    frd_files = sorted(Path(data_folder).glob("*.frd"))
    if not frd_files:
        raise FileNotFoundError(f"No .frd files found in {data_folder}")
    data = pd.concat([read_frd(f) for f in frd_files], ignore_index=True)

    rows = []
    for sensor, d in data.groupby("sensor"):
        rows.append({
            "sensor":          sensor,
            "analyte":         d.loc[d.step_type == "ASSOC", "sample_id"].iloc[0],
            "analyte_conc_nM": d.loc[d.step_type == "ASSOC", "concentration_nM"].iloc[0],
            "ligand":          d.loc[d.step_type == "LOADING", "sample_id"].iloc[0],
        })
    per_sensor = pd.DataFrame(rows)
    # A sensor "loaded" from a plain buffer well carries no ligand -> reference sensor
    # (measures non-specific binding / bulk shift of the analyte on the bare streptavidin tip).
    per_sensor["ligand_loaded"] = per_sensor["ligand"].str.len() > 0
    return data.merge(per_sensor, on="sensor")


def sensor_table(data):
    """One row per sensor: file, sensor, ligand, analyte, concentration."""
    return (data.drop_duplicates("sensor")
            [["file", "sensor", "ligand", "analyte", "analyte_conc_nM", "ligand_loaded"]]
            .sort_values("file")
            .reset_index(drop=True))


# ============================================
# PRE-PROCESSING
# ============================================
def align_kinetic_steps(data, baseline_window_s=5.0):
    """
    Cut out association + dissociation and align them:
      time = 0 at the start of association,
      response = 0 at the mean of the last baseline_window_s [s] of the preceding baseline.
    """
    aligned = []
    for _, d in data.groupby("sensor"):
        assoc_start_s = d.loc[d.step_type == "ASSOC", "time_s"].min()  # [s]
        baseline = d[(d.step_type == "BASELINE") & (d.time_s < assoc_start_s)
                     & (d.time_s >= assoc_start_s - baseline_window_s)]
        baseline_level_nm = baseline["response_nm"].mean()  # [nm]
        kinetic = d[d.step_type.isin(["ASSOC", "DISASSOC"])].copy()
        kinetic["time_aligned_s"]      = kinetic["time_s"] - assoc_start_s
        kinetic["response_aligned_nm"] = kinetic["response_nm"] - baseline_level_nm
        aligned.append(kinetic)
    return pd.concat(aligned, ignore_index=True)


def reference_subtract(aligned, mode="double"):
    """
    Build one referenced association/dissociation trace per (analyte, concentration)
    from the ligand-loaded sensors.

    mode:
      "none"      : loaded sensor only
      "reference" : loaded - reference sensor at the same analyte concentration
                    (removes non-specific binding to bare SA and the bulk refractive-index shift)
      "zero"      : loaded - loaded sensor at 0 nM analyte (removes drift of the loaded tip)
      "double"    : (loaded - reference) - (loaded_0nM - reference_0nM)   [standard double referencing]

    Returns a long DataFrame with columns
      analyte, analyte_conc_nM, sensor, step_type, time_aligned_s, response_nm
    The 0 nM trace is dropped for "zero" and "double" (it is identically zero there).
    """
    if mode not in ("none", "reference", "zero", "double"):
        raise ValueError(f"Unknown referencing mode '{mode}'")

    def trace(d, analyte, conc, loaded):
        s = d[(d.analyte == analyte) & (d.analyte_conc_nM == conc) & (d.ligand_loaded == loaded)]
        if s.sensor.nunique() != 1:
            raise ValueError(f"Expected exactly one sensor for {analyte}, {conc} nM, loaded={loaded}")
        return s

    def on_grid(s, time_grid_s):
        # Sensors share the step timing, but interpolate in case the grids differ by a sample
        return np.interp(time_grid_s, s.time_aligned_s.to_numpy(), s.response_aligned_nm.to_numpy())

    out = []
    loaded = aligned[aligned.ligand_loaded]
    for analyte in sorted(loaded.analyte.unique()):
        concs = sorted(loaded.loc[loaded.analyte == analyte, "analyte_conc_nM"].unique())
        for conc in concs:
            if mode in ("zero", "double") and conc == 0:
                continue
            main = trace(aligned, analyte, conc, True)
            t = main.time_aligned_s.to_numpy()
            y = main.response_aligned_nm.to_numpy().copy()
            if mode in ("reference", "double"):
                y -= on_grid(trace(aligned, analyte, conc, False), t)
            if mode in ("zero", "double"):
                y_zero = on_grid(trace(aligned, analyte, 0.0, True), t)
                if mode == "double":
                    y_zero -= on_grid(trace(aligned, analyte, 0.0, False), t)
                y -= y_zero
            out.append(pd.DataFrame({
                "analyte":         analyte,
                "analyte_conc_nM": conc,
                "sensor":          main.sensor.iloc[0],
                "step_type":       main.step_type.to_numpy(),
                "time_aligned_s":  t,
                "response_nm":     y,
            }))
    return pd.concat(out, ignore_index=True)


def dissociation_start_s(referenced):
    """Time [s] (aligned) at which dissociation starts; identical for all sensors of a run."""
    return referenced.loc[referenced.step_type == "DISASSOC", "time_aligned_s"].min()


# ============================================
# PER-CURVE ASSOCIATION FIT (single exponential)
# ============================================
def association_exponential(time_s, k_obs, R_eq, R_0):
    """
    Pseudo-first-order association of a 1:1 interaction at constant analyte concentration C:
        R(t) = R_eq + (R_0 - R_eq) * exp(-k_obs * t),   k_obs = k_on * C + k_off
    time_s [s], k_obs [1/s], R_eq [nm], R_0 [nm] -> R [nm]
    """
    return R_eq + (R_0 - R_eq) * np.exp(-k_obs * time_s)


def fit_association_exponential(time_s, response_nm):
    """
    Fit association_exponential to one association phase.
    k_obs is fitted as log10(k_obs) so it stays positive and spans decades without bound tuning.
    Returns dict(k_obs_per_s, R_eq_nm, R_0_nm, rmse_nm).
    """
    from scipy.optimize import least_squares

    duration_s = time_s[-1] - time_s[0]  # [s]

    def residuals(p):
        return association_exponential(time_s, 10.0 ** p[0], p[1], p[2]) - response_nm

    # Start: time constant = 1/10 of the step, plateau = mean of the last 5 % of the step
    tail_nm = response_nm[time_s >= time_s[0] + 0.95 * duration_s].mean()
    start   = [np.log10(10.0 / duration_s), tail_nm, response_nm[0]]
    result  = least_squares(residuals, start, method="trf")  # unbounded trust-region least squares
    return {
        "k_obs_per_s": 10.0 ** result.x[0],
        "R_eq_nm":     result.x[1],
        "R_0_nm":      result.x[2],
        "rmse_nm":     np.sqrt(np.mean(result.fun ** 2)),
    }


# ============================================
# PLOTTING HELPERS
# ============================================
STEP_COLORS = {"BASELINE": "0.93", "LOADING": "#dbe9f6", "ASSOC": "#e3f1df", "DISASSOC": "#fbe8d9"}


def concentration_colors(concentrations_nM):
    """Map concentrations to a sequential colormap on a log scale (0 nM gets grey)."""
    cmap = plt.get_cmap("viridis")
    positive = sorted(c for c in concentrations_nM if c > 0)
    log_min, log_max = np.log10(positive[0]), np.log10(positive[-1])
    colors = {}
    for c in concentrations_nM:
        colors[c] = "0.5" if c <= 0 else cmap(0.9 * (np.log10(c) - log_min) / max(log_max - log_min, 1e-12))
    return colors
