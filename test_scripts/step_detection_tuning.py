"""
Step Detection Tuning Script
-------------------------------
Loads a single event trace from the untrimmed HDF5 file and tests
different ruptures Pelt penalty values + cost models, plotting the
detected breakpoints overlaid on the trace so you can visually tune
the parameters before applying them to the full pipeline.

Requirements:
    pip install h5py numpy pandas matplotlib ruptures
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

# ── Configuration ─────────────────────────────────────────────────────────────

H5_PATH  = "./data/data_for_plots/data_with_real_current/aLA_holo_1_1/data100_peakandsteps_exstended_filtered.h5"   # path to untrimmed HDF5 file
CSV_PATH = "./data/data_for_plots/data_with_real_current/aLA_holo_1_1/data100_peakandsteps_exstended_filtered.csv"  # path to untrimmed CSV

EVENT_NAME = "event_00054"   # event to test step detection on
SAMPLING_RATE_KHZ = 50       # sampling rate in kHz

# values to try — edit these lists to experiment
COST_MODELS = ["l2", "rbf", "normal"]       # cost models to test
PENALTIES   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]  # penalty values to test

# ── Load trace and metadata ───────────────────────────────────────────────────

meta = pd.read_csv(CSV_PATH, index_col="event_name")
row  = meta.loc[EVENT_NAME]

with h5py.File(H5_PATH, "r") as f:
    trace = np.array(f[f"events/{EVENT_NAME}"], dtype=np.float64)

start = int(row["start"])
end   = int(row["end"])
dwell_samples = end - start
post_buffer   = int(round(dwell_samples / 3))
step_end      = min(end + post_buffer, len(trace))

step_region = trace[start:step_end]   # region to run step detection on
time_axis   = np.arange(len(step_region)) / SAMPLING_RATE_KHZ  # time axis in ms

print(f"Event: {EVENT_NAME}")
print(f"Step region: {len(step_region)} samples ({len(step_region)/SAMPLING_RATE_KHZ:.2f} ms)")

# ── Try different cost models and penalties ──────────────────────────────────

n_rows = len(COST_MODELS)
n_cols = len(PENALTIES)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)

for i, model in enumerate(COST_MODELS):
    for j, penalty in enumerate(PENALTIES):
        ax = axes[i, j]

        algo        = rpt.Pelt(model=model).fit(step_region)
        breakpoints = algo.predict(pen=penalty)
        n_steps     = max(0, len(breakpoints) - 1)

        ax.plot(time_axis, step_region, color="steelblue", linewidth=0.7)
        for bp in breakpoints[:-1]:  # last breakpoint is the trace length, skip it
            ax.axvline(time_axis[bp], color="red", linestyle="--", linewidth=1)

        ax.set_title(f"{model}, pen={penalty} → {n_steps} steps", fontsize=9)
        if i == n_rows - 1:
            ax.set_xlabel("Time (ms)")
        if j == 0:
            ax.set_ylabel("Current (nA)")

plt.suptitle(f"Step Detection Tuning — {EVENT_NAME}", fontsize=14)
plt.tight_layout()
plt.savefig("test_scripts_plots/step_detection_tuning.png", dpi=150)
plt.show()
print("Saved: test_scripts_plots/step_detection_tuning.png")