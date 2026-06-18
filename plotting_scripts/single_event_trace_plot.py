"""
Single Event Trace Plot
-------------------------
Loads a single event trace from an HDF5 file and plots it. Two modes:
  - SHOW_SCALE_BAR = False: normal labeled axes (Time / Current).
  - SHOW_SCALE_BAR = True:  no axes at all, just the trace plus an
    L-shaped scale bar (vertical = current, horizontal = time).
    The scale bar size can either be auto-computed (a "nice" round
    number based on the trace's range) or a fixed size you choose,
    controlled by SCALE_BAR_MODE.

Requirements:
    pip install h5py numpy pandas matplotlib
"""

import h5py                        # lets us open and read .h5 (HDF5) files, which is how the raw traces are stored
import numpy as np                 # used for array math (e.g. finding min/max, building the time axis)
import pandas as pd                # used to load the CSV file that has the event metadata (start, end, etc.)
import matplotlib.pyplot as plt    # used to draw and save the plot
from pathlib import Path           # used to create the output folder if it doesn't exist yet

# ── Configuration ─────────────────────────────────────────────────────────────

H5_PATH  = "./data/data_for_plots/data_with_real_current/4_events/data100_forstepsimage_filtered.h5"   # path to the HDF5 file with the raw traces
CSV_PATH = "./data/data_for_plots/data_with_real_current/4_events/data100_forstepsimage_filtered.csv"  # path to the CSV file with the event metadata

EVENT_NAME = "event_00054"   # which event (by name) to plot
SAMPLING_RATE_KHZ = 50       # how many samples per millisecond the recording has (used to convert samples to time)

PRE_BUFFER_FRACTION  = 1/3   # how much baseline to show before the event starts, as a fraction of the event's own duration
POST_BUFFER_FRACTION = 1/3   # how much baseline to show after the event ends, as a fraction of the event's own duration

SHOW_SCALE_BAR = True   # True: hide the normal axes and draw a scale bar instead. False: show normal labeled axes.

SCALE_BAR_MODE = "fixed"   # "auto": size the bars automatically based on the trace. "fixed": use the exact sizes set below.
FIXED_BAR_X = 25     # used only when SCALE_BAR_MODE = "fixed": length of the horizontal (time) bar, in ms
FIXED_BAR_Y = 0.1   # used only when SCALE_BAR_MODE = "fixed": length of the vertical (current) bar, in nA

SCALE_BAR_Y_FRACTION = 0.30   # used only when SCALE_BAR_MODE = "auto": target length of the vertical bar, as a fraction of the trace's current range
SCALE_BAR_X_FRACTION = 0.15   # used only when SCALE_BAR_MODE = "auto": target length of the horizontal bar, as a fraction of the trace's time range (kept smaller so it stays clear of the dip)

SCALE_BAR_X_MARGIN_FRAC = 0.05   # how far the scale bar sits from the left edge, as a fraction of the time range
SCALE_BAR_Y_MARGIN_FRAC = 0.05   # how far the scale bar sits below the trace, as a fraction of the current range (kept small so the bar sits close to the trace and the image stays compact)

TRACE_COLOR         = "black"   # color of the trace line
TRACE_LINEWIDTH     = 1.8       # thickness of the trace line (thicker so it stays visible once the image is shrunk)
SCALE_BAR_LINEWIDTH = 2.5       # thickness of the scale bar lines
SCALE_BAR_FONTSIZE  = 28        # font size of the scale bar numbers (e.g. "5 ms")

FIGURE_WIDTH_PX  = 1200   # width of the saved PNG, in pixels
FIGURE_HEIGHT_PX = 500   # height of the saved PNG, in pixels
DPI = 200                # resolution used to convert the pixel size above into inches, which is the unit matplotlib actually uses internally

OUTPUT_PNG = f"trace_plots/{EVENT_NAME}_trace.png"   # where the finished plot image will be saved

# ── Load trace and metadata ───────────────────────────────────────────────────

meta = pd.read_csv(CSV_PATH, index_col="event_name")   # load the metadata CSV and use the event name as the row label
row  = meta.loc[EVENT_NAME]                              # pick out the metadata row for the event we want to plot

with h5py.File(H5_PATH, "r") as f:                                     # open the HDF5 file for reading
    trace = np.array(f[f"events/{EVENT_NAME}"], dtype=np.float64)      # load the full raw trace for this event as a numpy array

start = int(row["start"])           # sample index where the event starts
end   = int(row["end"])             # sample index where the event ends
dwell_samples = end - start         # how many samples long the event itself is

pre_buffer  = int(round(dwell_samples * PRE_BUFFER_FRACTION))    # how many baseline samples to add before the event start
post_buffer = int(round(dwell_samples * POST_BUFFER_FRACTION))   # how many baseline samples to add after the event end

plot_start = max(start - pre_buffer, 0)              # first sample index to plot (never below 0)
plot_end   = min(end + post_buffer, len(trace))      # last sample index to plot (never past the end of the trace)

plot_region = trace[plot_start:plot_end]                          # the slice of the trace we'll actually plot
time_axis   = np.arange(len(plot_region)) / SAMPLING_RATE_KHZ     # convert sample indices to milliseconds, starting at 0

print(f"Event: {EVENT_NAME}")                                                                     # show which event we're plotting
print(f"Plot region: {len(plot_region)} samples ({len(plot_region)/SAMPLING_RATE_KHZ:.2f} ms)")    # show how long the plotted region is

# ── Scale bar helpers ──────────────────────────────────────────────────────────

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
    """Draw an L-shaped scale bar (vertical=current, horizontal=time) and hide the normal axes."""
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

    ax.text(anchor_x - margin_x * 0.3, anchor_y, f"{bar_y:g} nA",          # add the current label, rotated, starting at the bottom of the vertical bar so it grows upward, not outward from the middle
        rotation=90, ha="right", va="bottom", fontsize=fontsize)
    ax.text(anchor_x, anchor_y - margin_y * 0.9, f"{bar_x:g} ms",          # add the time label below the horizontal bar, starting at its left end so it grows rightward, not outward from the middle
        ha="left", va="top", fontsize=fontsize)

    ax.set_xlim(time_axis[0] - margin_x, time_axis[-1] + margin_x)    # adjust the visible x-range so the bar fits nicely
    ax.set_ylim(anchor_y - margin_y, trace_values.max() + margin_y)   # adjust the visible y-range so the bar fits nicely

    ax.set_xticks([])                              # remove the x-axis tick marks
    ax.set_yticks([])                               # remove the y-axis tick marks
    for spine in ax.spines.values():                # loop over the four box lines that normally frame the plot
        spine.set_visible(False)                    # hide each one, since we don't want a normal axis box

    print(f"Scale bar: {bar_y:g} nA, {bar_x:g} ms")    # print the final bar sizes, useful for sanity-checking

# ── Plot ────────────────────────────────────────────────────────────────────────

fig_width_in  = FIGURE_WIDTH_PX / DPI    # convert the requested pixel width into inches, since that's the unit matplotlib sizes figures in
fig_height_in = FIGURE_HEIGHT_PX / DPI   # convert the requested pixel height into inches, same reason as above

fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in))   # create a new figure and axes, sized to match the requested pixel dimensions

ax.plot(time_axis, plot_region, color=TRACE_COLOR, linewidth=TRACE_LINEWIDTH)   # draw the trace itself

if SHOW_SCALE_BAR:                                  # check whether we want the scale-bar style
    add_scale_bar(                                  # draw the scale bar and strip the normal axes
        ax, time_axis, plot_region,                 # the plot and the data it should be sized from
        SCALE_BAR_MODE, FIXED_BAR_X, FIXED_BAR_Y,    # auto vs fixed, and the fixed sizes if used
        SCALE_BAR_X_FRACTION, SCALE_BAR_Y_FRACTION,  # auto-size targets if used
        SCALE_BAR_X_MARGIN_FRAC, SCALE_BAR_Y_MARGIN_FRAC,  # how far the bar sits from the trace
        SCALE_BAR_LINEWIDTH, SCALE_BAR_FONTSIZE,     # how thick the bars are and how big the numbers are
    )
else:                                               # otherwise, fall back to a normal labeled plot
    ax.set_xlabel("Time (ms)")                      # label the x-axis
    ax.set_ylabel("Current (nA)")                   # label the y-axis
    ax.set_title(f"Event: {EVENT_NAME}")             # add a title with the event name

Path(OUTPUT_PNG).parent.mkdir(parents=True, exist_ok=True)   # create the output folder if it doesn't already exist
plt.tight_layout()                                            # adjust spacing so nothing gets cut off
plt.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")         # save the figure to disk; bbox_inches="tight" keeps the scale bar labels from being clipped, but can make the final pixel size slightly different from FIGURE_WIDTH_PX/HEIGHT_PX
plt.show()                                                     # display the plot on screen
print(f"Saved: {OUTPUT_PNG}")                                  # confirm where the file was saved