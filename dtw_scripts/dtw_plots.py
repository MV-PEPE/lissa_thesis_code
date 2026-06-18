"""
DTW Plotting Script
---------------------
Loads precomputed DTW results from an .npz file (produced by
dtw_analysis.py) and generates the DTW-aligned overlay plot and the
raw baseline-subtracted overlay plot.

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

INPUT_NPZ = "dtw_plots/dtw_results.npz"  # path to the saved DTW results

DTW_PLOT_CUTOFF_MS = 100  # x-axis cutoff for the DTW-aligned overlay plot
RAW_PLOT_CUTOFF_MS = 110  # x-axis cutoff for the raw baseline-subtracted overlay plot
MEAN_TRACE_CUTOFF_MS = 100  # x-axis cutoff for the mean aligned trace plot

STEP_WINDOW    = 40  # window size for std-based step detection (in samples)
STEP_THRESHOLD = 5   # t-statistic threshold for std-based step detection
# ── Load precomputed results ──────────────────────────────────────────────────

data = np.load(INPUT_NPZ, allow_pickle=True)

aligned      = data["aligned"]
time_aligned = data["time_aligned"]
ref_signal   = data["ref_signal"]
mean_trace   = data["mean_trace"]
rolling_std  = data["rolling_std"]
raw_signals  = data["raw_signals"]
raw_times    = data["raw_times"]
ref_key      = str(data["ref_key"])
# ref_dwell    = float(data["ref_dwell"])

# ── Plot 1: DTW-aligned overlay ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

for trace in aligned:
    ax.plot(time_aligned, trace, color="steelblue", alpha=0.15, linewidth=0.5)

ax.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5, label="Mean aligned trace")
ax.plot(time_aligned, rolling_std, color="orange", linewidth=1.5, label="Rolling std of mean")
ax.plot(time_aligned, ref_signal, color="black", linewidth=1, linestyle="--", label="Reference")

ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlim(time_aligned[0], DTW_PLOT_CUTOFF_MS)  # limit x-axis to the configured cutoff
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Baseline-subtracted current (nA)")
ax.set_title("DTW-Aligned Event Overlays")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("dtw_plots/dtw_aligned_overlay.png", dpi=150)
plt.show()
print("Saved: dtw_plots/dtw_aligned_overlay.png")

# ── Plot 2: Raw baseline-subtracted overlay (no DTW alignment) ────────────────

fig2, ax2 = plt.subplots(figsize=(12, 6))

for sig, time_axis in zip(raw_signals, raw_times):
    ax2.plot(time_axis, sig, color="steelblue", alpha=0.15, linewidth=0.5)

ax2.set_xlim(0, RAW_PLOT_CUTOFF_MS)  # limit x-axis to the configured cutoff
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Baseline-subtracted current (nA)")
ax2.set_title("Raw Event Overlays (baseline-subtracted, no alignment)")
plt.tight_layout()
plt.savefig("dtw_plots/raw_baseline_overlay.png", dpi=150)
plt.show()
print("Saved: dtw_plots/raw_baseline_overlay.png")

# ── Plot 3: Mean aligned trace alone ──────────────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(12, 6))

ax3.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5, label="Mean aligned trace")

ax3.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax3.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)  # limit x-axis to the configured cutoff
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Baseline-subtracted current (nA)")
ax3.set_title("Mean Aligned Trace")
ax3.legend(loc="lower right")
plt.tight_layout()
plt.savefig("dtw_plots/mean_aligned_trace.png", dpi=150)
plt.show()
print("Saved: dtw_plots/mean_aligned_trace.png")

# ── Plot 4: Step detection on the mean aligned trace ──────────────────────────

def detect_steps(signal, window, threshold):
    """Slide two adjacent windows, compute t-statistic-like score, flag step locations."""
    n      = len(signal)
    scores = np.zeros(n)

    for i in range(window, n - window):
        before = signal[i - window : i]
        after  = signal[i : i + window]

        mean_diff  = after.mean() - before.mean()
        pooled_std = np.sqrt((before.var() + after.var()) / 2)

        if pooled_std > 0:
            scores[i] = mean_diff / pooled_std

    flagged = np.where(np.abs(scores) > threshold)[0]

    steps = []
    if len(flagged) > 0:
        group_start = flagged[0]
        for k in range(1, len(flagged)):
            if flagged[k] != flagged[k-1] + 1:
                steps.append((group_start + flagged[k-1]) // 2)
                group_start = flagged[k]
        steps.append((group_start + flagged[-1]) // 2)

    return steps

peak_idx   = np.argmin(mean_trace)                                 # index of the deepest dip in the mean aligned trace
cutoff_idx = np.searchsorted(time_aligned, MEAN_TRACE_CUTOFF_MS)   # index corresponding to the plot's cutoff time
step_region = mean_trace[peak_idx:cutoff_idx]                      # region from dip to the cutoff

steps   = detect_steps(step_region, STEP_WINDOW, STEP_THRESHOLD)
n_steps = len(steps)
print(f"Detected {n_steps} steps in mean aligned trace")

fig4, ax4 = plt.subplots(figsize=(12, 6))

ax4.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5)

for step_idx in steps:
    ax4.axvline(time_aligned[peak_idx + step_idx], color="red", linestyle="--", linewidth=1)

ax4.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Baseline-subtracted current (nA)")
ax4.set_title(f"Step Detection on Mean Aligned Trace ({n_steps} steps detected)")
plt.tight_layout()
plt.savefig("dtw_plots/mean_trace_step_detection.png", dpi=150)
plt.show()
print("Saved: dtw_plots/mean_trace_step_detection.png")
