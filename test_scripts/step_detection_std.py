"""
Std-Based (T-Statistic) Step Detection Tuning Script
-------------------------------------------------------
Loads a single event trace and slides two adjacent windows across it,
computing a t-statistic-like score: the difference in means between
the windows, normalised by the pooled standard deviation. Flags a
step where this score exceeds a threshold. Plots the trace with
detected step points for different window sizes and thresholds.

Requirements:
    pip install h5py numpy pandas matplotlib
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

H5_PATH  = "./data/data_for_plots/data_with_real_current/4_events/data100_forstepsimage_filtered.h5"   # path to untrimmed HDF5 file
CSV_PATH = "./data/data_for_plots/data_with_real_current/4_events/data100_forstepsimage_filtered.csv"  # path to untrimmed CSV

EVENT_NAME = "event_00054"   # event to test step detection on
SAMPLING_RATE_KHZ = 50       # sampling rate in kHz

# values to try — edit these lists to experiment
WINDOW_SIZES = [5, 10, 20, 30]    # size of each comparison window (in samples)
THRESHOLDS   = [2, 3, 4, 5]       # t-statistic threshold

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

peak_idx     = start + np.argmin(trace[start:end])  # find index of maximum current dip within the event
step_region  = trace[peak_idx:step_end]             # region to run step detection on, starting at the peak dip
time_axis   = np.arange(len(step_region)) / SAMPLING_RATE_KHZ  # time axis in ms

print(f"Event: {EVENT_NAME}")
print(f"Step region: {len(step_region)} samples ({len(step_region)/SAMPLING_RATE_KHZ:.2f} ms)")

# ── Step detection function ───────────────────────────────────────────────────

def detect_steps(signal, window, threshold):
    """Slide two adjacent windows, compute t-statistic-like score, flag steps."""
    n      = len(signal)
    scores = np.zeros(n)  # t-statistic score at each point

    for i in range(window, n - window):                              # avoid edges where windows don't fit
        before = signal[i - window : i]                              # window before this point
        after  = signal[i : i + window]                              # window after this point

        mean_diff = after.mean() - before.mean()                     # difference in means
        pooled_std = np.sqrt((before.var() + after.var()) / 2)       # pooled standard deviation

        if pooled_std > 0:                                            # avoid division by zero
            scores[i] = mean_diff / pooled_std                        # t-statistic-like score
        else:
            scores[i] = 0

    flagged = np.where(np.abs(scores) > threshold)[0]                 # indices exceeding threshold

    # group consecutive flagged indices into single step events
    steps = []
    if len(flagged) > 0:
        group_start = flagged[0]
        for k in range(1, len(flagged)):
            if flagged[k] != flagged[k-1] + 1:                        # gap found, end current group
                steps.append((group_start + flagged[k-1]) // 2)       # use midpoint of group as step location
                group_start = flagged[k]
        steps.append((group_start + flagged[-1]) // 2)                # append final group

    return scores, steps

# ── Try different windows and thresholds ──────────────────────────────────────

n_rows = len(WINDOW_SIZES)
n_cols = len(THRESHOLDS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)

for i, window in enumerate(WINDOW_SIZES):
    for j, threshold in enumerate(THRESHOLDS):
        ax = axes[i, j]

        scores, steps = detect_steps(step_region, window, threshold)
        n_steps = len(steps)

        ax.plot(time_axis, step_region, color="steelblue", linewidth=0.7)  # raw trace
        for step_idx in steps:
            ax.axvline(time_axis[step_idx], color="red", linestyle="--", linewidth=1)  # detected step

        ax.set_title(f"window={window}, thresh={threshold} → {n_steps} steps", fontsize=9)
        if i == n_rows - 1:
            ax.set_xlabel("Time (ms)")
        if j == 0:
            ax.set_ylabel("Current (nA)")

plt.suptitle(f"Std-Based Step Detection Tuning — {EVENT_NAME}", fontsize=14)
plt.tight_layout()
plt.savefig("test_scripts_plots/step_detection_std_tuning.png", dpi=150)
plt.show()
print("Saved: test_scripts_plots/step_detection_std_tuning.png")