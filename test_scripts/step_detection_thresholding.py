"""
Derivative Thresholding Step Detection Tuning Script
-------------------------------------------------------
Loads a single event trace, smooths it with a rolling mean, computes
the derivative, and flags points where the derivative magnitude
exceeds a threshold (in units of the derivative's std). Plots the
trace with detected step points for different smoothing windows and
thresholds so you can visually tune the parameters.

Requirements:
    pip install h5py numpy pandas matplotlib
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

H5_PATH  = "./data/data_for_plots/data_with_real_current/aLA_holo_1_1/data100_peakandsteps_exstended_filtered.h5"   # path to untrimmed HDF5 file
CSV_PATH = "./data/data_for_plots/data_with_real_current/aLA_holo_1_1/data100_peakandsteps_exstended_filtered.csv"  # path to untrimmed CSV

EVENT_NAME = "event_00054"   # event to test step detection on
SAMPLING_RATE_KHZ = 50       # sampling rate in kHz

# values to try — edit these lists to experiment
SMOOTH_WINDOWS = [5, 10, 20, 30]      # rolling mean window sizes (in samples)
THRESHOLDS     = [1.5, 2, 3, 4]       # threshold in units of derivative std

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

# ── Step detection function ───────────────────────────────────────────────────

def detect_steps(signal, window, threshold):
    """Smooth signal, compute derivative, flag points exceeding threshold * std."""
    smoothed   = pd.Series(signal).rolling(window=window, center=True, min_periods=1).mean().values  # smooth signal
    derivative = np.diff(smoothed)                                    # first derivative (point-to-point difference)
    thresh_val = threshold * np.std(derivative)                       # threshold in units of derivative std

    flagged = np.where(np.abs(derivative) > thresh_val)[0]            # indices where derivative exceeds threshold

    # group consecutive flagged indices into single step events
    steps = []
    if len(flagged) > 0:
        group_start = flagged[0]
        for k in range(1, len(flagged)):
            if flagged[k] != flagged[k-1] + 1:                        # gap found, end current group
                steps.append((group_start + flagged[k-1]) // 2)       # use midpoint of group as step location
                group_start = flagged[k]
        steps.append((group_start + flagged[-1]) // 2)                # append final group

    return smoothed, derivative, steps

# ── Try different windows and thresholds ──────────────────────────────────────

n_rows = len(SMOOTH_WINDOWS)
n_cols = len(THRESHOLDS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)

for i, window in enumerate(SMOOTH_WINDOWS):
    for j, threshold in enumerate(THRESHOLDS):
        ax = axes[i, j]

        smoothed, derivative, steps = detect_steps(step_region, window, threshold)
        n_steps = len(steps)

        ax.plot(time_axis, step_region, color="lightgray", linewidth=0.5, label="raw")  # raw trace
        ax.plot(time_axis, smoothed, color="steelblue", linewidth=1, label="smoothed")  # smoothed trace
        for step_idx in steps:
            ax.axvline(time_axis[step_idx], color="red", linestyle="--", linewidth=1)  # detected step

        ax.set_title(f"window={window}, thresh={threshold} → {n_steps} steps", fontsize=9)
        if i == n_rows - 1:
            ax.set_xlabel("Time (ms)")
        if j == 0:
            ax.set_ylabel("Current (nA)")

plt.suptitle(f"Derivative Step Detection Tuning — {EVENT_NAME}", fontsize=14)
plt.tight_layout()
plt.savefig("test_scripts_plots/step_detection_derivative_tuning.png", dpi=150)
plt.show()
print("Saved: test_scripts_plots/step_detection_derivative_tuning.png")