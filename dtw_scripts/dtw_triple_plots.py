"""
DTW Triple Overlay Plot
-------------------------
Loads three precomputed DTW results from three separate .npz files
(produced by dtw_analysis.py) and generates a single figure with
three side-by-side DTW-aligned overlay panels, one per file.
Each panel has its own independent x-axis cutoff.
A shared legend is placed below all three panels.

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

# each entry is (path to .npz file, x-axis cutoff in ms, panel title)
PANELS = [
    (
        "./data/data_for_plots/dtw_analysis_results/aLA_apo_1_1_event_00248_dtw_results.npz",  # path to the first DTW results file
        75,                                                                    # x-axis cutoff for this panel, in ms
        "(a)",                                                              # title shown above this panel
    ),
    (
        "./data/data_for_plots/dtw_analysis_results/aLA_apo_1_1_event_00170_dtw_results.npz",  # path to the second DTW results file
        130,                                                                    # x-axis cutoff for this panel, in ms
        "(b)",                                                              # title shown above this panel
    ),
    (
        "./data/data_for_plots/dtw_analysis_results/aLA_apo_1_1_event_00282_dtw_results.npz",  # path to the third DTW results file
        80,                                                                    # x-axis cutoff for this panel, in ms
        "(c)",                                                              # title shown above this panel
    ),
]

OVERLAY_COLOR   = "steelblue"   # color of the individual aligned traces shown faintly behind the mean
OVERLAY_ALPHA   = 0.15          # transparency of each individual trace; lower = more see-through
OVERLAY_WIDTH   = 0.5           # line thickness of each individual trace
MEAN_COLOR      = "darkred"     # color of the mean aligned trace line
MEAN_WIDTH      = 1.5           # thickness of the mean aligned trace line
STD_COLOR       = "orange"      # color of the rolling std line
STD_WIDTH       = 1.5           # thickness of the rolling std line
REF_COLOR       = "black"       # color of the reference trace line
REF_WIDTH       = 1.0           # thickness of the reference trace line

FONT_SIZE       = 30            # base font size for all text: axis labels, tick numbers, legend, and panel titles
LEGEND_Y        = -0.15         # vertical position of the legend, relative to the bottom of the panels (negative = below; more negative = further down)

OUTPUT_PNG = "dtw_plots/dtw_triple_overlay.png"   # path to save the combined figure

# ── Build figure ──────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(24, 8))   # one row, three columns; each panel gets the same width

for ax, (npz_path, cutoff_ms, title) in zip(axes, PANELS):   # loop over each panel and its config

    # ── Load this panel's data ────────────────────────────────────────────────

    data         = np.load(npz_path, allow_pickle=True)   # load the .npz file produced by dtw_analysis.py
    aligned      = data["aligned"]                         # DTW-aligned traces, shape (n_events, n_samples)
    time_aligned = data["time_aligned"]                    # time axis for the aligned traces, in ms
    ref_signal   = data["ref_signal"]                      # reference trace used for DTW alignment
    mean_trace   = data["mean_trace"]                      # mean of all aligned traces
    rolling_std  = data["rolling_std"]                     # rolling std of the mean trace

    # ── Draw this panel ───────────────────────────────────────────────────────

    for trace in aligned:   # loop over each individual aligned trace
        ax.plot(time_aligned, trace, color=OVERLAY_COLOR, alpha=OVERLAY_ALPHA, linewidth=OVERLAY_WIDTH)   # plot faintly behind the mean

    ax.plot(time_aligned, mean_trace, color=MEAN_COLOR, linewidth=MEAN_WIDTH, label="Mean aligned trace")     # mean trace on top
    ax.plot(time_aligned, rolling_std, color=STD_COLOR,  linewidth=STD_WIDTH,  label="Rolling std of mean")   # rolling std
    ax.plot(time_aligned, ref_signal,  color=REF_COLOR,  linewidth=REF_WIDTH,  label="Reference", linestyle="--")  # reference trace

    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)               # vertical line at t=0
    ax.set_xlim(time_aligned[0], cutoff_ms)                                 # apply this panel's own x-axis cutoff
    ax.set_xlabel("Time (ms)", fontsize=FONT_SIZE)                          # x-axis label
    ax.set_ylabel("Baseline-subtracted\ncurrent (nA)", fontsize=FONT_SIZE)   # y-axis label
    ax.set_title(title, fontsize=FONT_SIZE + 2, pad=20)                             # panel title from config
    ax.tick_params(axis="both", labelsize=FONT_SIZE)                        # tick numbers on both axes

# ── Shared legend below all three panels ──────────────────────────────────────

handles, labels = axes[0].get_legend_handles_labels()       # get the legend entries from the first panel (same for all three)
fig.legend(                                                 # place a single shared legend for the whole figure
    handles, labels,                                        # the line handles and their labels
    loc="lower center",                                     # anchor point of the legend box
    ncol=3,                                                 # put all three legend entries side by side in one row
    fontsize=FONT_SIZE,                                     # legend text size
    bbox_to_anchor=(0.5, LEGEND_Y),                         # fine-tune the position: centered horizontally, LEGEND_Y below the panels
)

n_events = aligned.shape[0]                                 # number of events — same for all three panels since they share the same DTW run; taken from the last loaded panel
fig.suptitle(f"     {n_events} events\n", fontsize=34)             # single title above all three panels showing the event count
plt.tight_layout()                                          # adjust spacing between panels so nothing overlaps
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")       # bbox_inches="tight" ensures the legend below doesn't get clipped
plt.show()
print(f"Saved: {OUTPUT_PNG}")