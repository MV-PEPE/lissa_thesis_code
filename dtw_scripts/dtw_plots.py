"""
DTW Plotting Script
---------------------
Loads precomputed DTW results from an .npz file (produced by
dtw_analysis.py) and generates 5 separate plots:
  1. DTW-aligned overlay
  2. Raw baseline-subtracted overlay
  3. Mean aligned trace alone
  4. Step detection on the mean aligned trace
  5. Staircase (piecewise constant) approximation of the mean aligned trace

Config values and the step-detection algorithm match dtw_plots_grid.py.

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

INPUT_NPZ = "dtw_plots/dtw_results.npz"  # path to the saved DTW results

DTW_PLOT_CUTOFF_MS   = 110  # x-axis cutoff for the DTW-aligned overlay plot
RAW_PLOT_CUTOFF_MS   = 60   # x-axis cutoff for the raw baseline-subtracted overlay plot
RAW_PLOT_PRE_DIP_MS  = RAW_PLOT_CUTOFF_MS / 2  # how much time to show before the dip; this is also where the dip ends up on the x-axis, since x=0 is now the left edge
MEAN_TRACE_CUTOFF_MS = 110  # x-axis cutoff for the mean aligned trace plots

# how many samples on either side of a point to compare when checking for a step.
# SMALLER = more sensitive to short/sharp steps but noisier, can split one real step into two.
# LARGER = smoother and less prone to noise, but can blur out small/quick steps or
# merge two close-together real steps into one
STEP_WINDOW = 100  # window size for std-based step detection (in samples)

# how big a jump has to be (in standard deviations) to count as a step.
# LOWER = catches smaller/subtler steps but risks false positives from noise.
# HIGHER = only catches large, clear jumps but may miss subtle ones
STEP_THRESHOLD = 4  # t-statistic threshold for std-based step detection

# how far before the dip step detection should start looking, in ms (0 = start exactly at the dip)
STEP_DETECTION_PRE_DIP_MS = 40

# if two detected steps land closer together than this (in ms), merge them into one.
# Fixes a single sharp transition occasionally being detected as two separate steps
MIN_STEP_SEPARATION_MS = 2

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

# ── Step detection (runs once, shared by the step-detection and staircase plots) ──

def detect_steps(signal, window, threshold, min_separation=0):
    """Slide two adjacent windows, compute t-statistic-like score, flag step locations.
    Any two detected steps closer together than min_separation (in samples) get merged
    into one, fixing a single sharp transition that occasionally gets split into two."""
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

    if min_separation > 0 and len(steps) > 1:               # merge any steps that ended up too close together
        merged = [steps[0]]
        for step in steps[1:]:
            if step - merged[-1] < min_separation:          # this step is too close to the last kept one
                merged[-1] = (merged[-1] + step) // 2       # replace it with the midpoint of the two
            else:
                merged.append(step)                         # far enough away, keep it as its own step
        steps = merged

    return steps

peak_idx   = np.argmin(mean_trace)                                 # index of the deepest dip in the mean aligned trace
cutoff_idx = np.searchsorted(time_aligned, MEAN_TRACE_CUTOFF_MS)   # index corresponding to the plot's cutoff time

step_start_time = time_aligned[peak_idx] - STEP_DETECTION_PRE_DIP_MS      # how far before the dip step detection should start, in ms
step_start      = max(np.searchsorted(time_aligned, step_start_time), 0)  # corresponding index; clamped so it never goes before the very start of the trace
step_region     = mean_trace[step_start:cutoff_idx]                       # region to run step detection on, possibly starting before the dip

ms_per_sample           = time_aligned[1] - time_aligned[0]               # time spacing between consecutive samples, in ms
min_separation_samples  = round(MIN_STEP_SEPARATION_MS / ms_per_sample)   # convert the configured ms separation into samples, since detect_steps works in samples

steps = detect_steps(step_region, STEP_WINDOW, STEP_THRESHOLD, min_separation_samples)   # run step detection
steps = [s for s in steps if abs((step_start + s) - peak_idx) >= min_separation_samples]  # drop any detected step that's really just the dip transition itself; the dip gets its own line below, independent of this list

n_steps = len(steps)                                                # number of detected steps (not counting the dip itself)
print(f"Detected {n_steps} steps in mean aligned trace")

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
    peak_idx_raw = np.argmin(sig)                                    # index of this event's own peak dip
    time_shifted = time_axis - time_axis[peak_idx_raw] + RAW_PLOT_PRE_DIP_MS  # align dips across traces, but keep x=0 at the plot's left edge, not at the dip
    ax2.plot(time_shifted, sig, color="steelblue", alpha=0.15, linewidth=0.5)

ax2.set_xlim(0, RAW_PLOT_PRE_DIP_MS + RAW_PLOT_CUTOFF_MS)  # start at the left edge (x=0), dip sits at RAW_PLOT_PRE_DIP_MS, cutoff extends RAW_PLOT_CUTOFF_MS past the dip
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

fig4, ax4 = plt.subplots(figsize=(12, 6))

ax4.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5)

ax4.axvline(time_aligned[peak_idx], color="red", linestyle="--", linewidth=1)  # mark the dip itself as a step too, same style as the others

for step_idx in steps:
    ax4.axvline(time_aligned[step_start + step_idx], color="red", linestyle="--", linewidth=1)

ax4.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)
ax4.set_xlabel("Time (ms)")
ax4.set_ylabel("Baseline-subtracted current (nA)")
ax4.set_title(f"Step Detection on Mean Aligned Trace ({n_steps + 1} steps detected)")  # +1 for the dip itself
plt.tight_layout()
plt.savefig("dtw_plots/mean_trace_step_detection.png", dpi=150)
plt.show()
print("Saved: dtw_plots/mean_trace_step_detection.png")

# ── Plot 5: Staircase approximation of the mean aligned trace ─────────────────

# build the boundary indices that separate each flat segment:
# the trace is divided by the step positions, extended to the plot start and cutoff
step_boundaries = sorted(set(                                         # one index per boundary between flat segments
    [0]                                                               # start of the plot region
    + [peak_idx]                                                      # the dip itself, also treated as a step boundary
    + [step_start + s for s in steps]                                 # each detected step position, shifted back to full-trace indices
    + [cutoff_idx]                                                    # end of the plot region (at the cutoff)
))

fig5, ax5 = plt.subplots(figsize=(12, 6))

for i in range(len(step_boundaries) - 1):                             # loop over each segment between consecutive boundaries
    seg_start = step_boundaries[i]                                    # first sample index of this segment
    seg_end   = step_boundaries[i + 1]                                # last sample index of this segment (exclusive)
    seg_mean  = mean_trace[seg_start:seg_end].mean()                  # mean current level of the mean trace within this segment
    t_start   = time_aligned[seg_start]                               # time of the segment's left boundary
    t_end     = time_aligned[seg_end - 1]                             # time of the segment's right boundary
    segment_label = "Mean current level" if i == 0 else None          # only label the first segment, so the legend doesn't get one entry per flat line
    ax5.plot([t_start, t_end], [seg_mean, seg_mean],                  # draw a flat horizontal line at the segment's mean level
             color="darkred", linewidth=3, label=segment_label)

ax5.axvline(time_aligned[peak_idx], color="red", linestyle="--", linewidth=1, label="Detected step")  # mark the dip itself as a step too, same style as the others

for step_idx in steps:                                                # draw the same vertical step lines as in the step detection panel
    ax5.axvline(time_aligned[step_start + step_idx], color="red", linestyle="--", linewidth=1)

ax5.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)
ax5.set_xlabel("Time (ms)")
ax5.set_ylabel("Baseline-subtracted current (nA)")
ax5.set_title(f"Steps and Current Levels of Mean Aligned Trace ({n_steps + 1} unique levels detected)")  # +1 for the dip itself
ax5.legend(loc="lower right")
plt.tight_layout()
plt.savefig("dtw_plots/mean_trace_staircase.png", dpi=150)
plt.show()
print("Saved: dtw_plots/mean_trace_staircase.png")