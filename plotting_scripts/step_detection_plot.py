"""
Step Detection Plot (Std-Based Approach)
-------------------------------------------
Loads a single event trace, runs std-based (t-statistic) step
detection starting from the peak current dip, and generates two plots:
  1. The event trace itself, with detected step locations marked.
  2. A staircase (piecewise constant) approximation of the same event,
     showing one flat segment per detected current level.

Step-detection algorithm matches dtw_plots.py: optional pre-dip
lookback, merging of steps detected too close together, and the dip
itself counted as a step.

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

EVENT_NAME = "event_00516"   # event to plot
SAMPLING_RATE_KHZ = 50       # sampling rate in kHz (= samples per ms)

PRE_BUFFER_FRACTION  = 1/3   # how much baseline to show before the event starts, as a fraction of the event's own dwell time
POST_BUFFER_FRACTION = 1/3   # how much baseline to show after the event ends, as a fraction of the event's own dwell time

# how many samples on either side of a point to compare when checking for a step.
# SMALLER = more sensitive to short/sharp steps but noisier, can split one real step into two.
# LARGER = smoother and less prone to noise, but can blur out small/quick steps or
# merge two close-together real steps into one
STEP_WINDOW = 30  # window size for std-based step detection (in samples)

# how big a jump has to be (in standard deviations) to count as a step.
# LOWER = catches smaller/subtler steps but risks false positives from noise.
# HIGHER = only catches large, clear jumps but may miss subtle ones
STEP_THRESHOLD = 3.6  # t-statistic threshold for std-based step detection

# how far before the dip step detection should start looking, in ms (0 = start exactly at the dip, like before)
STEP_DETECTION_PRE_DIP_MS = 10

# if two detected steps land closer together than this (in ms), merge them into one.
# Fixes a single sharp transition occasionally being detected as two separate steps
MIN_STEP_SEPARATION_MS = 0.5

SCALE_BAR_MODE = "auto"   # "auto": size the scale bar automatically based on the trace. "fixed": use the exact sizes set below.
FIXED_BAR_X = 5     # used only when SCALE_BAR_MODE = "fixed": length of the horizontal (time) bar, in ms
FIXED_BAR_Y = 0.3   # used only when SCALE_BAR_MODE = "fixed": length of the vertical (current) bar, in nA

SCALE_BAR_Y_FRACTION = 0.60   # used only when SCALE_BAR_MODE = "auto": target length of the vertical bar, as a fraction of the trace's current range
SCALE_BAR_X_FRACTION = 0.15   # used only when SCALE_BAR_MODE = "auto": target length of the horizontal bar, as a fraction of the trace's time range

SCALE_BAR_X_MARGIN_FRAC = 0.02   # how far the scale bar sits from the left edge, as a fraction of the time range
SCALE_BAR_Y_MARGIN_FRAC = 0.1   # how far the scale bar sits below the trace, as a fraction of the current range

SCALE_BAR_LINEWIDTH = 4       # thickness of the scale bar lines
SCALE_BAR_FONTSIZE  = 60        # font size of the scale bar numbers (e.g. "5 ms")

TRACE_OUTPUT_PNG     = f"step_detection_plots/event_trace_{EVENT_NAME}.png"               # path to save the plain trace plot (no markers)
OUTPUT_PNG           = f"step_detection_plots/step_detection_plot_{EVENT_NAME}.png"       # path to save the trace + step lines plot
STAIRCASE_OUTPUT_PNG = f"step_detection_plots/step_detection_staircase_{EVENT_NAME}.png"  # path to save the staircase plot

# ── Load trace and metadata ───────────────────────────────────────────────────

meta = pd.read_csv(CSV_PATH, index_col="event_name")
row  = meta.loc[EVENT_NAME]

with h5py.File(H5_PATH, "r") as f:
    trace = np.array(f[f"events/{EVENT_NAME}"], dtype=np.float64)

start = int(row["start"])
end   = int(row["end"])
dwell_samples = end - start
pre_buffer    = int(round(dwell_samples * PRE_BUFFER_FRACTION))   # how many baseline samples to add before the event start
post_buffer   = int(round(dwell_samples * POST_BUFFER_FRACTION))  # how many baseline samples to add after the event end
plot_start    = max(start - pre_buffer, 0)             # first sample index to display (never below 0)
step_end      = min(end + post_buffer, len(trace))

peak_idx    = start + np.argmin(trace[start:end])     # index of the deepest current dip within the event
plot_region = trace[plot_start:step_end]               # full region to display, now including some baseline before the event start
time_axis   = np.arange(len(plot_region)) / SAMPLING_RATE_KHZ  # time axis for the displayed trace, starting at 0

pre_dip_samples = round(STEP_DETECTION_PRE_DIP_MS * SAMPLING_RATE_KHZ)        # how many samples before the dip to start looking, converted from ms
step_start      = max(peak_idx - pre_dip_samples, plot_start)                      # where step detection actually starts; clamped so it never goes before the displayed region
step_region     = trace[step_start:step_end]                                  # region to run step detection on, possibly starting before the dip

peak_offset_ms        = (peak_idx - plot_start) / SAMPLING_RATE_KHZ    # position of the dip within the displayed trace, in ms (used to mark the dip itself)
step_region_offset_ms = (step_start - plot_start) / SAMPLING_RATE_KHZ  # position of the step-detection region's start within the displayed trace, in ms (used as the base offset for detected steps)

print(f"Event: {EVENT_NAME}")
print(f"Step region: {len(step_region)} samples ({len(step_region)/SAMPLING_RATE_KHZ:.2f} ms)")

# ── Step detection function ───────────────────────────────────────────────────

def detect_steps(signal, window, threshold, min_separation=0):
    """Slide two adjacent windows, compute t-statistic-like score, flag step locations.
    Any two detected steps closer together than min_separation (in samples) get merged
    into one, fixing a single sharp transition that occasionally gets split into two."""
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

    if min_separation > 0 and len(steps) > 1:               # merge any steps that ended up too close together
        merged = [steps[0]]
        for step in steps[1:]:
            if step - merged[-1] < min_separation:          # this step is too close to the last kept one
                merged[-1] = (merged[-1] + step) // 2       # replace it with the midpoint of the two
            else:
                merged.append(step)                         # far enough away, keep it as its own step
        steps = merged

    return steps

# ── Detect steps ───────────────────────────────────────────────────────────────

min_separation_samples = round(MIN_STEP_SEPARATION_MS * SAMPLING_RATE_KHZ)  # convert the configured ms separation into samples

steps = detect_steps(step_region, STEP_WINDOW, STEP_THRESHOLD, min_separation_samples)   # run step detection
steps = [s for s in steps if abs((step_start + s) - peak_idx) >= min_separation_samples]  # drop any detected step that's really just the dip transition itself; the dip gets its own line below, independent of this list

n_steps = len(steps)  # number of detected steps (not counting the dip itself)
print(f"Detected {n_steps} steps")

# ── Scale bar helpers (ported from trace_plot.py) ──────────────────────────────

def nice_length(value):
    """Round value down to the nearest 'nice' number (1, 2, or 5 x 10^n)."""
    if value <= 0:                             # guard against zero or negative input
        return 0                               # nothing sensible to return in that case
    exponent = np.floor(np.log10(value))       # find the power of 10 that value falls into
    fraction = value / 10**exponent            # rescale value to a number between 1 and 10
    if fraction < 1.5:                         # decide which "nice" digit (1, 2, 5, or 10) is closest
        nice_fraction = 1                      # round to 1 x 10^exponent
    elif fraction < 3:
        nice_fraction = 2                      # round to 2 x 10^exponent
    elif fraction < 7:
        nice_fraction = 5                      # round to 5 x 10^exponent
    else:
        nice_fraction = 10                     # round to 10 x 10^exponent
    return nice_fraction * 10**exponent        # return the final "nice" number


def add_scale_bar(ax, time_axis, trace_values, mode, fixed_x, fixed_y, x_frac, y_frac, x_margin_frac, y_margin_frac, linewidth, fontsize):
    """Draw an L-shaped scale bar (vertical=current, horizontal=time). Unlike trace_plot.py's
    version, this one keeps the plot's box (spines) visible, since that's this script's style."""
    x_range = time_axis[-1] - time_axis[0]                       # total time span of the plotted trace
    y_range = trace_values.max() - trace_values.min()            # total current range of the plotted trace

    if mode == "fixed":                            # use the fixed sizes passed in from the config section
        bar_x = fixed_x                            # horizontal bar length, taken directly from config
        bar_y = fixed_y                             # vertical bar length, taken directly from config
    else:                                           # otherwise, auto-compute "nice" sizes from the data
        bar_x = nice_length(x_range * x_frac)       # horizontal bar length, auto-sized
        bar_y = nice_length(y_range * y_frac)       # vertical bar length, auto-sized

    margin_x = x_range * x_margin_frac            # how far from the left edge the bar starts
    margin_y = y_range * y_margin_frac            # how far below the trace the bar starts
    anchor_x = time_axis[0] + margin_x             # x-position of the bottom-left corner of the "L"
    anchor_y = trace_values.min() - margin_y       # y-position of the bottom-left corner of the "L"

    ax.plot([anchor_x, anchor_x], [anchor_y, anchor_y + bar_y], color="black", linewidth=linewidth)   # draw the vertical (current) bar
    ax.plot([anchor_x, anchor_x + bar_x], [anchor_y, anchor_y], color="black", linewidth=linewidth)    # draw the horizontal (time) bar

    ax.text(anchor_x - margin_x * 0.3, anchor_y, f"{bar_y:g} nA",          # add the current label, starting at the bottom of the vertical bar so it grows upward
            rotation=90, ha="right", va="bottom", fontsize=fontsize)
    ax.text(anchor_x, anchor_y - margin_y * 0.3, f"{bar_x:g} ms",          # add the time label below the horizontal bar
            ha="left", va="top", fontsize=fontsize)

    ax.set_xlim(time_axis[0] - margin_x, time_axis[-1] + margin_x)    # adjust the visible x-range so the bar fits nicely
    ax.set_ylim(anchor_y - margin_y, trace_values.max() + margin_y)   # adjust the visible y-range so the bar fits nicely

    # ax.set_xticks([])                              # remove the x-axis tick marks
    # ax.set_yticks([])                               # remove the y-axis tick marks
    # note: spines (the box) are intentionally left visible here, unlike trace_plot.py's version

    print(f"Scale bar: {bar_y:g} nA, {bar_x:g} ms")    # print the final bar sizes, useful for sanity-checking

# ── Plot 1: Simple event trace (no markers) ─────────────────────────────────────

fig0, ax0 = plt.subplots(figsize=(12, 6))

ax0.plot(time_axis, plot_region, color="black", linewidth=1.5)  # plot the full trace from event start, no step markers

ax0.set_xlim(time_axis[0], time_axis[-1])  # remove matplotlib's default auto-margin so the trace fills the plot edge to edge

plt.tight_layout()
plt.savefig(TRACE_OUTPUT_PNG, dpi=150)
plt.show()
print(f"Saved: {TRACE_OUTPUT_PNG}")

# ── Plot 2: Event trace with detected steps ─────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(time_axis, plot_region, color="black", linewidth=3)  # plot the full trace from event start

ax1.axvline(peak_offset_ms, color="green", linestyle="--", linewidth=2)  # mark the dip itself as a step too, same style as the others

for step_idx in steps:                                              # mark each detected step
    ax1.axvline(step_region_offset_ms + step_idx / SAMPLING_RATE_KHZ, color="green", linestyle="--", linewidth=2)  # shift step positions to match full trace

ax1.set_xlim(time_axis[0], time_axis[-1])  # remove matplotlib's default auto-margin so the trace fills the plot edge to edge
ax1.set_xticks([])  # remove the x-axis tick numbers
ax1.set_yticks([])  # remove the y-axis tick numbers

add_scale_bar(
    ax1, time_axis, plot_region,                         # the plot and the data it should be sized from
    SCALE_BAR_MODE, FIXED_BAR_X, FIXED_BAR_Y,             # auto vs fixed, and the fixed sizes if used
    SCALE_BAR_X_FRACTION, SCALE_BAR_Y_FRACTION,           # auto-size targets if used
    SCALE_BAR_X_MARGIN_FRAC, SCALE_BAR_Y_MARGIN_FRAC,     # how far the bar sits from the trace
    SCALE_BAR_LINEWIDTH, SCALE_BAR_FONTSIZE,              # how thick the bars are and how big the numbers are
)

for spine in ax1.spines.values():    # loop over the four box lines that normally frame the plot
    spine.set_visible(False)         # hide each one, so no box is drawn around this plot

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150)
plt.show()
print(f"Saved: {OUTPUT_PNG}")

# ── Build staircase boundaries ─────────────────────────────────────────────────

peak_idx_relative = peak_idx - plot_start  # the dip's position within plot_region (plot_region starts at 'start', not at the dip)

step_boundaries = sorted(set(                                       # one index per boundary between flat segments, in plot_region-relative indices; sorted + deduped since pre-dip lookback can put a step before the dip
    [0]                                                              # start of the displayed region
    + [peak_idx_relative]                                            # the dip itself, also treated as a step boundary
    + [(step_start - plot_start) + step_idx for step_idx in steps]        # each detected step position, shifted to plot_region-relative indices
    + [len(plot_region) - 1]                                         # end of the displayed region
))

# ── Plot 3: Staircase approximation of the event ────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(12, 6))

for i in range(len(step_boundaries) - 1):                          # loop over each segment between consecutive boundaries
    seg_start = step_boundaries[i]                                  # first sample index of this segment
    seg_end   = step_boundaries[i + 1]                               # last sample index of this segment (exclusive)
    seg_mean  = plot_region[seg_start:seg_end].mean()                # mean current level within this segment
    t_start   = time_axis[seg_start]                                 # time of the segment's left boundary
    t_end     = time_axis[seg_end - 1]                               # time of the segment's right boundary
    ax2.plot([t_start, t_end], [seg_mean, seg_mean], color="green", linewidth=5)  # flat horizontal line at the segment's mean level

ax2.axvline(peak_offset_ms, color="green", linestyle="--", linewidth=2)  # mark the dip itself as a step too, same style as the others

for step_idx in steps:                                              # mark each detected step, same as in the trace plot
    ax2.axvline(step_region_offset_ms + step_idx / SAMPLING_RATE_KHZ, color="green", linestyle="--", linewidth=2)

ax2.set_xlim(time_axis[0], time_axis[-1])  # remove matplotlib's default auto-margin so the staircase fills the plot edge to edge
ax2.set_xticks([])  # remove the x-axis tick numbers
ax2.set_yticks([])  # remove the y-axis tick numbers

for spine in ax2.spines.values():    # loop over the four box lines that normally frame the plot
    spine.set_visible(False)         # hide each one, so no box is drawn around this plot

ax2.set_title(f"#{n_steps} unique levels", fontsize=60, color="green", y=-0.15)

plt.tight_layout()
plt.savefig(STAIRCASE_OUTPUT_PNG, dpi=150)
plt.show()
print(f"Saved: {STAIRCASE_OUTPUT_PNG}")