"""
DTW Plotting Script
---------------------
Loads precomputed DTW results from an .npz file (produced by
dtw_analysis.py) and generates all 4 plots arranged in a 2x2 grid:
  top-left:     Raw baseline-subtracted overlay
  top-right:    DTW-aligned overlay
  bottom-left:  Mean aligned trace alone
  bottom-right: Mean aligned trace with step detection

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

INPUT_NPZ = "dtw_plots/dtw_results.npz"  # path to the saved DTW results

DTW_PLOT_CUTOFF_MS  = 140  # x-axis cutoff for the DTW-aligned overlay plot
RAW_PLOT_CUTOFF_MS  = 110  # x-axis cutoff for the raw baseline-subtracted overlay plot
MEAN_TRACE_CUTOFF_MS = 140  # x-axis cutoff for the mean aligned trace plots

STEP_WINDOW    = 50  # window size for std-based step detection (in samples)
STEP_THRESHOLD = 5   # t-statistic threshold for std-based step detection

OUTPUT_PNG = "dtw_plots/dtw_combined_2x2.png"  # path to save the combined 2x2 figure

# ── Load precomputed results ──────────────────────────────────────────────────

data = np.load(INPUT_NPZ, allow_pickle=True)

aligned      = data["aligned"]        # DTW-aligned traces
time_aligned = data["time_aligned"]   # time axis for the aligned traces
ref_signal   = data["ref_signal"]     # reference trace used for DTW alignment
mean_trace   = data["mean_trace"]     # mean of all aligned traces
rolling_std  = data["rolling_std"]    # rolling std of the mean trace
raw_signals  = data["raw_signals"]    # raw baseline-subtracted traces
raw_times    = data["raw_times"]      # per-event time axes for the raw traces
ref_key      = str(data["ref_key"])   # name of the reference event

# ── Step detection (runs before plotting so n_steps is available for the title) ──

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

peak_idx    = np.argmin(mean_trace)                                # index of the deepest dip in the mean aligned trace
cutoff_idx  = np.searchsorted(time_aligned, MEAN_TRACE_CUTOFF_MS)  # index corresponding to the plot's cutoff time
step_region = mean_trace[peak_idx:cutoff_idx]                      # region from dip to the cutoff to run step detection on

steps   = detect_steps(step_region, STEP_WINDOW, STEP_THRESHOLD)   # run step detection
n_steps = len(steps)                                                # number of detected steps
print(f"Detected {n_steps} steps in mean aligned trace")

# ── Combined 2x2 figure ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(20, 10))  # 2 rows, 2 columns; each subplot gets the same space

ax_raw        = axes[0, 0]  # top-left:     raw baseline-subtracted overlay
ax_dtw        = axes[0, 1]  # top-right:    DTW-aligned overlay
ax_steps      = axes[1, 0]  # bottom-left:  mean aligned trace with step detection
ax_staircase  = axes[1, 1]  # bottom-right: piecewise constant (staircase) approximation of the mean trace

# ── Top-left: Raw baseline-subtracted overlay ─────────────────────────────────

for sig, time_axis in zip(raw_signals, raw_times):                  # loop over each raw event trace
    peak_idx_raw  = np.argmin(sig)                                  # index of this event's own peak dip
    time_shifted  = time_axis - time_axis[peak_idx_raw]             # shift time axis so peak dip = t=0
    ax_raw.plot(time_shifted, sig, color="steelblue", alpha=0.15, linewidth=0.5)  # plot the shifted trace

ax_raw.set_xlim(-RAW_PLOT_CUTOFF_MS / 4, RAW_PLOT_CUTOFF_MS)       # show a bit before the dip and up to the cutoff after
ax_raw.set_xlabel("Time (ms)")                                       # x-axis label
ax_raw.set_ylabel("Baseline-subtracted current (nA)")                # y-axis label
ax_raw.set_title("Raw Event Overlays (baseline-subtracted, no alignment)")  # subplot title

# ── Top-right: DTW-aligned overlay ───────────────────────────────────────────

for trace in aligned:                                                # loop over each DTW-aligned trace
    ax_dtw.plot(time_aligned, trace, color="steelblue", alpha=0.15, linewidth=0.5)  # plot faintly so the mean stands out

ax_dtw.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5, label="Mean aligned trace")   # mean trace on top
ax_dtw.plot(time_aligned, rolling_std, color="orange", linewidth=1.5, label="Rolling std of mean")  # rolling std
ax_dtw.plot(time_aligned, ref_signal, color="black", linewidth=1, linestyle="--", label="Reference")  # reference trace

ax_dtw.axvline(0, color="gray", linestyle=":", linewidth=0.8)       # vertical line at t=0
ax_dtw.set_xlim(time_aligned[0], DTW_PLOT_CUTOFF_MS)                # limit x-axis to the configured cutoff
ax_dtw.set_xlabel("Time (ms)")                                       # x-axis label
ax_dtw.set_ylabel("Baseline-subtracted current (nA)")                # y-axis label
ax_dtw.set_title("DTW-Aligned Event Overlays")                       # subplot title
ax_dtw.legend(loc="lower right")                                     # legend in the bottom-right corner

# ── Bottom-left: Mean aligned trace with step detection ─────────────────────

ax_steps.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5)  # mean trace

for step_idx in steps:                                               # mark each detected step with a vertical dashed line
    ax_steps.axvline(time_aligned[peak_idx + step_idx], color="red", linestyle="--", linewidth=1)

ax_steps.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)            # limit x-axis to the configured cutoff
ax_steps.set_xlabel("Time (ms)")                                     # x-axis label
ax_steps.set_ylabel("Baseline-subtracted current (nA)")              # y-axis label
ax_steps.set_title(f"Step Detection on Mean Aligned Trace ({n_steps} steps detected)")  # subplot title includes step count

# ── Bottom-right: Staircase approximation of the mean aligned trace ───────────

# build the boundary indices that separate each flat segment:
# the trace is divided by the step positions, extended to the plot start and cutoff
step_boundaries = (                                                   # one index per boundary between flat segments
    [0]                                                               # start of the plot region
    + [peak_idx + s for s in steps]                                  # each detected step position, shifted back to full-trace indices
    + [cutoff_idx]                                                    # end of the plot region (at the cutoff)
)

for i in range(len(step_boundaries) - 1):                            # loop over each segment between consecutive boundaries
    seg_start = step_boundaries[i]                                    # first sample index of this segment
    seg_end   = step_boundaries[i + 1]                                # last sample index of this segment (exclusive)
    seg_mean  = mean_trace[seg_start:seg_end].mean()                  # mean current level of the mean trace within this segment
    t_start   = time_aligned[seg_start]                               # time of the segment's left boundary
    t_end     = time_aligned[seg_end - 1]                             # time of the segment's right boundary
    ax_staircase.plot([t_start, t_end], [seg_mean, seg_mean],         # draw a flat horizontal line at the segment's mean level
                      color="darkred", linewidth=1.5)

for step_idx in steps:                                                # draw the same vertical step lines as in the step detection panel
    ax_staircase.axvline(time_aligned[peak_idx + step_idx], color="red", linestyle="--", linewidth=1)

ax_staircase.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)         # limit x-axis to the configured cutoff
ax_staircase.set_xlabel("Time (ms)")                                  # x-axis label
ax_staircase.set_ylabel("Baseline-subtracted current (nA)")           # y-axis label
ax_staircase.set_title(f"Staircase Approximation ({n_steps} steps)")  # subplot title

# ── Save ──────────────────────────────────────────────────────────────────────

n_events = len(aligned)                     # total number of DTW-aligned events, used for the figure title
fig.suptitle(f"{n_events} events", fontsize=16)  # single title above the whole 2x2 grid

plt.tight_layout()                          # adjust spacing between subplots so nothing overlaps
plt.savefig(OUTPUT_PNG, dpi=150)            # save the combined figure
plt.show()                                  # display on screen
print(f"Saved: {OUTPUT_PNG}")               # confirm where the file was saved