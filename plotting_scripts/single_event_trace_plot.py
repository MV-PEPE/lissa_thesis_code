"""
Single Event Trace Plot
-------------------------
Loads a single event trace from an HDF5 file and plots it. Two modes:
  - SHOW_SCALE_BAR = False: normal labeled axes (Time / Current).
  - SHOW_SCALE_BAR = True:  no axes at all, just the trace plus an
    L-shaped scale bar (vertical = current, horizontal = time), sized
    automatically to a "nice" round number based on the trace's range.

Requirements:
    pip install h5py numpy pandas matplotlib
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

H5_PATH  = "./data/data_for_plots/data_with_real_current/4_events/data100_forstepsimage_filtered.h5"
CSV_PATH = "./data/data_for_plots/data_with_real_current/4_events/data100_forstepsimage_filtered.csv"

EVENT_NAME = "event_00054"   # event to plot
SAMPLING_RATE_KHZ = 50       # sampling rate in kHz

PRE_BUFFER_FRACTION  = 1/3   # baseline shown before event start, as a fraction of dwell time
POST_BUFFER_FRACTION = 1/3   # baseline shown after event end, as a fraction of dwell time

SHOW_SCALE_BAR     = True    # True: no axes, auto scale bar. False: normal labeled axes.
SCALE_BAR_FRACTION = 0.3     # target scale bar length, as a fraction of the trace's data range (auto-rounded to a "nice" number)
SCALE_BAR_X_MARGIN_FRAC = 0.05  # left margin for the scale bar, as a fraction of the x-range
SCALE_BAR_Y_MARGIN_FRAC = 0.15  # margin below the trace for the scale bar, as a fraction of the y-range

TRACE_COLOR     = "black"
TRACE_LINEWIDTH = 0.8

OUTPUT_PNG = f"trace_plots/{EVENT_NAME}_trace.png"

# ── Load trace and metadata ───────────────────────────────────────────────────

meta = pd.read_csv(CSV_PATH, index_col="event_name")
row  = meta.loc[EVENT_NAME]

with h5py.File(H5_PATH, "r") as f:
    trace = np.array(f[f"events/{EVENT_NAME}"], dtype=np.float64)

start = int(row["start"])
end   = int(row["end"])
dwell_samples = end - start

pre_buffer  = int(round(dwell_samples * PRE_BUFFER_FRACTION))   # baseline samples to show before start
post_buffer = int(round(dwell_samples * POST_BUFFER_FRACTION))  # baseline samples to show after end

plot_start = max(start - pre_buffer, 0)
plot_end   = min(end + post_buffer, len(trace))

plot_region = trace[plot_start:plot_end]
time_axis   = np.arange(len(plot_region)) / SAMPLING_RATE_KHZ  # ms, relative to plot_start

print(f"Event: {EVENT_NAME}")
print(f"Plot region: {len(plot_region)} samples ({len(plot_region)/SAMPLING_RATE_KHZ:.2f} ms)")

# ── Scale bar helpers ──────────────────────────────────────────────────────────

def nice_length(value):
    """Round value down to the nearest 'nice' number (1, 2, or 5 x 10^n)."""
    if value <= 0:
        return 0
    exponent = np.floor(np.log10(value))
    fraction = value / 10**exponent
    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3:
        nice_fraction = 2
    elif fraction < 7:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * 10**exponent


def add_scale_bar(ax, time_axis, trace_values, frac, x_margin_frac, y_margin_frac, x_unit="ms", y_unit="nA"):
    """Draw an L-shaped scale bar (vertical=current, horizontal=time), auto-sized
    to a 'nice' fraction of the trace's data range, and hide the normal axes."""
    x_range = time_axis[-1] - time_axis[0]
    y_range = trace_values.max() - trace_values.min()

    bar_x = nice_length(x_range * frac)  # horizontal bar length, in x_unit
    bar_y = nice_length(y_range * frac)  # vertical bar length, in y_unit

    # anchor point: bottom-left corner of the "L", with margin below/left of the trace
    margin_x = x_range * x_margin_frac
    margin_y = y_range * y_margin_frac
    anchor_x = time_axis[0] + margin_x
    anchor_y = trace_values.min() - margin_y

    ax.plot([anchor_x, anchor_x], [anchor_y, anchor_y + bar_y], color="black", linewidth=2)  # vertical bar
    ax.plot([anchor_x, anchor_x + bar_x], [anchor_y, anchor_y], color="black", linewidth=2)  # horizontal bar

    ax.text(anchor_x - margin_x * 0.3, anchor_y + bar_y / 2, f"{bar_y:g} {y_unit}",
            rotation=90, ha="right", va="center", fontsize=11)
    ax.text(anchor_x + bar_x / 2, anchor_y - margin_y * 0.3, f"{bar_x:g} {x_unit}",
            ha="center", va="top", fontsize=11)

    ax.set_xlim(time_axis[0] - margin_x, time_axis[-1] + margin_x)
    ax.set_ylim(anchor_y - margin_y, trace_values.max() + margin_y)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    print(f"Scale bar: {bar_y:g} {y_unit}, {bar_x:g} {x_unit}")

# ── Plot ────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(time_axis, plot_region, color=TRACE_COLOR, linewidth=TRACE_LINEWIDTH)

if SHOW_SCALE_BAR:
    add_scale_bar(ax, time_axis, plot_region, SCALE_BAR_FRACTION, SCALE_BAR_X_MARGIN_FRAC, SCALE_BAR_Y_MARGIN_FRAC)
else:
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (nA)")
    ax.set_title(f"Event: {EVENT_NAME}")

Path(OUTPUT_PNG).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")  # bbox_inches="tight" avoids clipping the scale bar labels, which sit outside the axes
plt.show()
print(f"Saved: {OUTPUT_PNG}")