"""
Step Detection Plot (Std-Based Approach)
-------------------------------------------
Loads a single event trace, runs std-based (t-statistic) step
detection starting from the peak current dip, and plots the trace
with detected step locations marked.

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

EVENT_NAME = "event_00054"   # event to plot
SAMPLING_RATE_KHZ = 50       # sampling rate in kHz

STEP_WINDOW    = 20  # window size for std-based step detection (in samples)
STEP_THRESHOLD = 4   # t-statistic threshold for std-based step detection

OUTPUT_PNG = "dtw_plots/step_detection_plot.png"  # path to save the plot

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
plot_region  = trace[start:step_end]                 # full region to display, starting from event start
step_region  = trace[peak_idx:step_end]              # region to run step detection on, starting at the peak dip
time_axis    = np.arange(len(plot_region)) / SAMPLING_RATE_KHZ        # time axis for the displayed trace
step_offset_ms = (peak_idx - start) / SAMPLING_RATE_KHZ              # offset of peak dip from start, in ms

print(f"Event: {EVENT_NAME}")
print(f"Step region: {len(step_region)} samples ({len(step_region)/SAMPLING_RATE_KHZ:.2f} ms)")

# ── Step detection function ───────────────────────────────────────────────────

def detect_steps(signal, window, threshold):
    """Slide two adjacent windows, compute t-statistic-like score, flag step locations."""
    n      = len(signal)
    scores = np.zeros(n)  # t-statistic score at each point

    for i in range(window, n - window):                              # avoid edges where windows don't fit
        before = signal[i - window : i]                              # window before this point
        after  = signal[i : i + window]                              # window after this point

        mean_diff  = after.mean() - before.mean()                    # difference in means
        pooled_std = np.sqrt((before.var() + after.var()) / 2)       # pooled standard deviation

        if pooled_std > 0:                                            # avoid division by zero
            scores[i] = mean_diff / pooled_std                        # t-statistic-like score

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

    return steps

# ── Detect steps ───────────────────────────────────────────────────────────────

steps   = detect_steps(step_region, STEP_WINDOW, STEP_THRESHOLD)
n_steps = len(steps)
print(f"Detected {n_steps} steps")

# ── Plot ────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(time_axis, plot_region, color="steelblue", linewidth=0.8)  # plot the full trace from event start

for step_idx in steps:                                              # mark each detected step
    ax.axvline(step_offset_ms + step_idx / SAMPLING_RATE_KHZ, color="red", linestyle="--", linewidth=1)  # shift step positions to match full trace

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Current (nA)")
ax.set_title(f"Step Detection — {EVENT_NAME} ({n_steps} steps detected)")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"Saved: {OUTPUT_PNG}")