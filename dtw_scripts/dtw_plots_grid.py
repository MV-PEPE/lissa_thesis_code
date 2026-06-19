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

DTW_PLOT_CUTOFF_MS  = 110  # x-axis cutoff for the DTW-aligned overlay plot
RAW_PLOT_CUTOFF_MS  = 60  # x-axis cutoff for the raw baseline-subtracted overlay plot
RAW_PLOT_PRE_DIP_MS = RAW_PLOT_CUTOFF_MS / 2  # how much time to show before the dip; this is also where the dip ends up on the x-axis, since x=0 is now the left edge
MEAN_TRACE_CUTOFF_MS = 110  # x-axis cutoff for the mean aligned trace plots

# how many samples on either side of a point to compare when checking for a step. 
# SMALLER = more sensitive to short/sharp steps but noisier, can split one real step into two. 
# LARGER = smoother and less prone to noise, but can blur out small/quick steps or 
# merge two close-together real steps into one
STEP_WINDOW    = 100  # window size for std-based step detection (in samples)

# how big a jump has to be (in standard deviations) to count as a step. 
# LOWER = catches smaller/subtler steps but risks false positives from noise. 
# HIGHER = only catches large, clear jumps but may miss subtle ones
STEP_THRESHOLD = 4   # t-statistic threshold for std-based step detection

# how far before the dip step detection should start looking, in ms (0 = start exactly at the dip)
STEP_DETECTION_PRE_DIP_MS = 40

# if two detected steps land closer together than this (in ms), merge them into one. 
# Fixes a single sharp transition occasionally being detected as two separate steps
MIN_STEP_SEPARATION_MS = 2 

SUBPLOT_PAD = 5.0  # extra gap specifically between the 4 subplots, in font-size units (matplotlib's default is about 1.08); higher = more space

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

peak_idx    = np.argmin(mean_trace)                                # index of the deepest dip in the mean aligned trace
cutoff_idx  = np.searchsorted(time_aligned, MEAN_TRACE_CUTOFF_MS)  # index corresponding to the plot's cutoff time

step_start_time = time_aligned[peak_idx] - STEP_DETECTION_PRE_DIP_MS  # how far before the dip step detection should start, in ms
step_start      = max(np.searchsorted(time_aligned, step_start_time), 0)  # corresponding index; clamped so it never goes before the very start of the trace
step_region     = mean_trace[step_start:cutoff_idx]                 # region to run step detection on, now possibly starting before the dip

ms_per_sample           = time_aligned[1] - time_aligned[0]               # time spacing between consecutive samples, in ms
min_separation_samples  = round(MIN_STEP_SEPARATION_MS / ms_per_sample)   # convert the configured ms separation into samples, since detect_steps works in samples

steps   = detect_steps(step_region, STEP_WINDOW, STEP_THRESHOLD, min_separation_samples)   # run step detection

steps = [s for s in steps if abs((step_start + s) - peak_idx) >= min_separation_samples]  # drop any detected step that's really just the dip transition itself; the dip already gets its own line below, independent of this list

n_steps = len(steps)                                                # number of detected steps
print(f"Detected {n_steps} steps in mean aligned trace")

# ── Combined 2x2 figure ───────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(20, 10))  # 2 rows, 2 columns; each subplot gets the same space

ax_raw        = axes[0, 0]  # top-left:     raw baseline-subtracted overlay
ax_dtw        = axes[0, 1]  # top-right:    DTW-aligned overlay
ax_steps      = axes[1, 0]  # bottom-left:  mean aligned trace with step detection
ax_staircase  = axes[1, 1]  # bottom-right: piecewise constant (staircase) approximation of the mean trace

TICK_FONTSIZE = 16  # size of the numbers along the x and y axes, for all 4 subplots
for ax in axes.flat:                                  # loop over all 4 subplots at once
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)  # set the size of the axis tick numbers

# ── Top-left: Raw baseline-subtracted overlay ─────────────────────────────────

for sig, time_axis in zip(raw_signals, raw_times):                  # loop over each raw event trace
    peak_idx_raw = np.argmin(sig)                                    # index of this event's own peak dip
    time_shifted = time_axis - time_axis[peak_idx_raw] + RAW_PLOT_PRE_DIP_MS  # align dips across traces, but keep x=0 at the plot's left edge, not at the dip
    ax_raw.plot(time_shifted, sig, color="steelblue", alpha=0.15, linewidth=0.5)  # plot the shifted trace

ax_raw.set_xlim(0, RAW_PLOT_PRE_DIP_MS + RAW_PLOT_CUTOFF_MS)         # start at the left edge (x=0), dip sits at RAW_PLOT_PRE_DIP_MS, cutoff extends RAW_PLOT_CUTOFF_MS past the dip
ax_raw.set_xlabel("Time (ms)", fontsize=20)                          # x-axis label
ax_raw.set_ylabel("Baseline-subtracted current (nA)", fontsize=18)   # y-axis label
ax_raw.set_title("Raw Event Overlays (baseline-subtracted, no alignment)", fontsize=22)  # subplot title

# ── Top-right: DTW-aligned overlay ───────────────────────────────────────────

for trace in aligned:                                                # loop over each DTW-aligned trace
    ax_dtw.plot(time_aligned, trace, color="steelblue", alpha=0.15, linewidth=0.5)  # plot faintly so the mean stands out

ax_dtw.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5, label="Mean aligned trace")   # mean trace on top
ax_dtw.plot(time_aligned, rolling_std, color="orange", linewidth=1.5, label="Rolling std of mean")  # rolling std
ax_dtw.plot(time_aligned, ref_signal, color="black", linewidth=1, linestyle="--", label="Reference")  # reference trace

ax_dtw.axvline(0, color="gray", linestyle=":", linewidth=0.8)        # vertical line at t=0
ax_dtw.set_xlim(time_aligned[0], DTW_PLOT_CUTOFF_MS)                 # limit x-axis to the configured cutoff
ax_dtw.set_xlabel("Time (ms)", fontsize=20)                          # x-axis label
ax_dtw.set_ylabel("Baseline-subtracted current (nA)", fontsize=18)   # y-axis label
ax_dtw.set_title("DTW-Aligned Event Overlays", fontsize=22)          # subplot title
ax_dtw.legend(loc="lower left", fontsize=14)                        # legend in the bottom-right corner

# ── Bottom-left: Mean aligned trace with step detection ─────────────────────

ax_steps.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5)  # mean trace

ax_steps.axvline(time_aligned[peak_idx], color="green", linestyle="--", linewidth=1)  # mark the dip itself as a step too, same style as the others

for step_idx in steps:                                               # mark each detected step with a vertical dashed line
    ax_steps.axvline(time_aligned[step_start + step_idx], color="green", linestyle="--", linewidth=1)

ax_steps.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)            # limit x-axis to the configured cutoff
ax_steps.set_xlabel("Time (ms)", fontsize=20)                                     # x-axis label
ax_steps.set_ylabel("Baseline-subtracted current (nA)", fontsize=18)              # y-axis label
ax_steps.set_title(f"Step Detection on Mean Aligned Trace\n({n_steps} steps detected)", fontsize=22)  # subplot title includes step count

# ── Bottom-right: Staircase approximation of the mean aligned trace ───────────

# build the boundary indices that separate each flat segment:
# the trace is divided by the step positions, extended to the plot start and cutoff
step_boundaries = sorted(set((                                                   # one index per boundary between flat segments
    [0]                                                               # start of the plot region
    + [peak_idx]                                                      # the dip itself, now also treated as a step boundary
    + [step_start + s for s in steps]                                   # each detected step position, shifted back to full-trace indices
    + [cutoff_idx]                                                    # end of the plot region (at the cutoff)
)))

for i in range(len(step_boundaries) - 1):                             # loop over each segment between consecutive boundaries
    seg_start = step_boundaries[i]                                    # first sample index of this segment
    seg_end   = step_boundaries[i + 1]                                # last sample index of this segment (exclusive)
    seg_mean  = mean_trace[seg_start:seg_end].mean()                  # mean current level of the mean trace within this segment
    t_start   = time_aligned[seg_start]                               # time of the segment's left boundary
    t_end     = time_aligned[seg_end - 1]                             # time of the segment's right boundary
    segment_label = "Mean current level" if i == 0 else None          # only label the first segment, so the legend doesn't get one entry per flat line
    ax_staircase.plot([t_start, t_end], [seg_mean, seg_mean],         # draw a flat horizontal line at the segment's mean level
                      color="green", linewidth=3, label=segment_label)

ax_staircase.axvline(time_aligned[peak_idx], color="green", linestyle="--", linewidth=1, label="Detected step")  # mark the dip itself as a step too, same style as the others

for step_idx in steps:      # draw the same vertical step lines as in the step detection panel
    ax_staircase.axvline(time_aligned[step_start + step_idx], color="green", linestyle="--", linewidth=1)

ax_staircase.set_xlim(time_aligned[0], MEAN_TRACE_CUTOFF_MS)              # limit x-axis to the configured cutoff
ax_staircase.set_xlabel("Time (ms)", fontsize=20)                         # x-axis label
ax_staircase.set_ylabel("Baseline-subtracted current (nA)", fontsize=18)  # y-axis label
ax_staircase.set_title(f"Steps and Current Levels of Mean Aligned Trace\n({n_steps} unique levels detected)", fontsize=22)  # subplot title
ax_staircase.legend(loc="lower left", fontsize=14)  # legend for the mean current level and detected step lines

# ── Save ──────────────────────────────────────────────────────────────────────

n_events = len(aligned)                     # total number of DTW-aligned events, used for the figure title
fig.suptitle(f"{n_events} events\n", fontsize=24)  # single title above the whole 2x2 grid

plt.tight_layout(h_pad=SUBPLOT_PAD, w_pad=SUBPLOT_PAD)  # adjust spacing between subplots so nothing overlaps
plt.savefig(OUTPUT_PNG, dpi=150)            # save the combined figure
plt.show()                                  # display on screen
print(f"Saved: {OUTPUT_PNG}")               # confirm where the file was saved