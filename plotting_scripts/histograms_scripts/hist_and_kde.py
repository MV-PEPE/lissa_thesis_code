"""
Unified Histogram Script
-------------------------
Loads event metadata from all CSVs in data_with_real_current and
data_with_recovered_current, groups them by folder prefix, and
generates interactive histograms in one of six modes:

    PLOT_MODE options:
        "overlaid_bars"          — overlaid bar histogram, y = count
        "separate_bars"          — separate subplot per group, y = count
        "overlaid_kde_density"   — overlaid KDE smooth curve, y = density
        "separate_kde_density"   — separate subplot per group, y = density
        "overlaid_kde_counts"    — overlaid KDE smooth curve, y = counts (scaled)
        "separate_kde_counts"    — separate subplot per group, y = counts (scaled)

Requirements:
    pip install pandas plotly scipy
"""

import pandas as pd                        # for loading and combining CSV metadata
import numpy as np                         # for numerical operations
import plotly.graph_objects as go          # for building interactive plots
from plotly.subplots import make_subplots  # for creating subplot grids
import plotly.io as pio                    # for saving interactive HTML files
from pathlib import Path                   # for navigating the folder structure
from scipy.stats import gaussian_kde       # for computing KDE curves
import math                                # for computing subplot grid dimensions

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIRS = [
    Path("data/data_for_plots/data_with_real_current"),       # directory with real current data
    Path("data/data_for_plots/data_with_recovered_current"),  # directory with recovered current data
]

PLOT_MODE  = "overlaid_kde_density"   # see docstring above for options
BIN_WIDTH  = 5                 # bin width for bar histograms (in x-axis units)
KDE_POINTS = 500               # number of points to evaluate KDE curve at

GROUPS = {
    "aLA_holo": "aLA holo",  # group display names keyed by folder prefix
    "BSA":      "BSA",
    "aLA_apo":  "aLA apo",
}

COLORS = {
    "aLA_holo": "#e6194b",  # red
    "BSA":      "#2698ba",  # cyan
    "aLA_apo":  "#33c110",  # green
}

OUTPUT_DIR = Path("histograms")  # directory to save the output HTML files

# ── Load and group CSVs ────────────────────────────────────────────────────────

group_data = {g: [] for g in GROUPS}  # dictionary to collect dataframes per group

for data_dir in DATA_DIRS:                               # loop over both data directories
    for csv_path in data_dir.rglob("*.csv"):             # recursively find all CSV files
        folder_name = csv_path.parent.name               # get the subfolder name
        for prefix in GROUPS:                            # check which group this folder belongs to
            if folder_name.startswith(prefix):           # match folder name to group prefix
                df = pd.read_csv(csv_path)               # load the CSV
                df["group"] = prefix                     # tag each row with its group name
                group_data[prefix].append(df)            # add to the group's list
                break                                    # stop checking prefixes once matched

group_dfs = {}
for prefix, dfs in group_data.items():                   # loop over each group
    if dfs:                                              # only process groups that have data
        group_dfs[prefix] = pd.concat(dfs, ignore_index=True)  # combine all CSVs for this group
        print(f"Group '{prefix}': {len(group_dfs[prefix])} total events")
    else:
        print(f"Group '{prefix}': no data found")        # warn if a group has no data

OUTPUT_DIR.mkdir(exist_ok=True)  # create output directory if it doesn't exist

# ── KDE helper ─────────────────────────────────────────────────────────────────

def compute_kde(data, n_points, scaled=False):
    """Compute KDE curve for a 1D array of data values.
    
    Args:
        data:     1D array of values
        n_points: number of points to evaluate the KDE at
        scaled:   if True, scale density to counts (density * n * bin_width)
    
    Returns:
        x_vals: x positions of the KDE curve
        y_vals: y values (density or counts depending on scaled)
    """
    kde      = gaussian_kde(data)                                      # fit KDE to data
    x_vals   = np.linspace(data.min(), data.max(), n_points)           # evenly spaced x points across data range
    y_vals   = kde(x_vals)                                             # evaluate KDE at each x point
    if scaled:                                                         # scale density to counts if requested
        bin_width = (data.max() - data.min()) / n_points               # approximate bin width for scaling
        y_vals    = y_vals * len(data) * bin_width                     # scale: density * n_events * bin_width
    return x_vals, y_vals

# ── Plotting functions ─────────────────────────────────────────────────────────

def make_overlaid(col_name, x_label, y_label, filename, mode):
    """Create an overlaid histogram or KDE plot for all groups."""
    fig = go.Figure()                                    # create empty figure

    for prefix, df in group_dfs.items():                 # loop over each group
        data = df[col_name].dropna().values              # get data column, drop NaN values

        if mode == "overlaid_bars":                      # bar histogram mode
            fig.add_trace(go.Histogram(
                x=data,                                  # data to bin
                name=GROUPS[prefix],                     # group display name
                marker_color=COLORS[prefix],             # group color
                marker=dict(line=dict(width=2)),         # step style outline
                xbins=dict(size=BIN_WIDTH),              # fixed bin width
            ))
            fig.update_layout(barmode="overlay")         # overlay all groups

        else:                                            # KDE mode
            scaled  = mode == "overlaid_kde_counts"      # scale to counts if requested
            x, y    = compute_kde(data, KDE_POINTS, scaled=scaled)  # compute KDE curve
            r, g, b = int(COLORS[prefix][1:3], 16), int(COLORS[prefix][3:5], 16), int(COLORS[prefix][5:7], 16)  # extract RGB components from hex
            fig.add_trace(go.Scatter(
                x=x, y=y,                                # KDE curve x and y values
                name=GROUPS[prefix],                     # group display name
                mode="lines",                            # line only, no markers
                line=dict(color=COLORS[prefix], width=2),  # group color and line width
                fill="tozeroy",                          # fill area under curve to zero
                fillcolor=f"rgba({r},{g},{b},0.15)",  # build rgba string with 15% opacity
            ))

    fig.update_layout(
        xaxis_title=x_label,    # x-axis label
        yaxis_title=y_label,    # y-axis label
        legend_title="Group",   # legend title
    )

    out_path = OUTPUT_DIR / filename                     # full output path
    pio.write_html(fig, str(out_path))                   # save interactive HTML file
    fig.show()                                           # open in browser
    print(f"Saved: {out_path}")                          # confirm the file was saved


def make_separate(col_name, x_label, y_label, filename, mode):
    """Create a figure with one subplot per group."""
    n_groups = len(group_dfs)                            # number of groups with data
    n_cols   = 2                                         # number of columns in subplot grid
    n_rows   = math.ceil(n_groups / n_cols)              # number of rows needed

    fig = make_subplots(
        rows=n_rows, cols=n_cols,                        # subplot grid dimensions
        subplot_titles=[GROUPS[p] for p in group_dfs],  # title each subplot with group name
    )

    for i, (prefix, df) in enumerate(group_dfs.items()):  # loop over each group
        row_idx = i // n_cols + 1                        # row index for this subplot (1-indexed)
        col_idx = i % n_cols + 1                         # column index for this subplot (1-indexed)
        data    = df[col_name].dropna().values           # get data column, drop NaN values

        if mode == "separate_bars":                      # bar histogram mode
            fig.add_trace(
                go.Histogram(
                    x=data,                              # data to bin
                    name=GROUPS[prefix],                 # group display name
                    marker_color=COLORS[prefix],         # group color
                    marker=dict(line=dict(width=2)),     # step style outline
                    xbins=dict(size=BIN_WIDTH),          # fixed bin width
                    showlegend=False,                    # hide legend since subplots have titles
                ),
                row=row_idx, col=col_idx,                # place in correct subplot position
            )

        else:                                            # KDE mode
            scaled = mode == "separate_kde_counts"       # scale to counts if requested
            x, y   = compute_kde(data, KDE_POINTS, scaled=scaled)  # compute KDE curve
            fig.add_trace(
                go.Scatter(
                    x=x, y=y,                            # KDE curve x and y values
                    name=GROUPS[prefix],                 # group display name
                    mode="lines",                        # line only, no markers
                    line=dict(color=COLORS[prefix], width=2),  # group color and line width
                    fill="tozeroy",                      # fill area under curve to zero
                    showlegend=False,                    # hide legend since subplots have titles
                ),
                row=row_idx, col=col_idx,                # place in correct subplot position
            )

        fig.update_xaxes(title_text=x_label, row=row_idx, col=col_idx)  # x-axis label per subplot
        fig.update_yaxes(title_text=y_label, row=row_idx, col=col_idx)  # y-axis label per subplot

    fig.update_layout(height=400 * n_rows)               # scale figure height to number of rows

    out_path = OUTPUT_DIR / filename                     # full output path
    pio.write_html(fig, str(out_path))                   # save interactive HTML file
    fig.show()                                           # open in browser
    print(f"Saved: {out_path}")                          # confirm the file was saved


# ── Mode dispatch ──────────────────────────────────────────────────────────────

# map each mode to its y-axis label and plotting function
MODE_CONFIG = {
    "overlaid_bars":         ("Count",   make_overlaid),
    "separate_bars":         ("Count",   make_separate),
    "overlaid_kde_density":  ("Density", make_overlaid),
    "separate_kde_density":  ("Density", make_separate),
    "overlaid_kde_counts":   ("Count",   make_overlaid),
    "separate_kde_counts":   ("Count",   make_separate),
}

if PLOT_MODE not in MODE_CONFIG:                         # validate PLOT_MODE value
    raise ValueError(f"Unknown PLOT_MODE '{PLOT_MODE}'. Choose from: {list(MODE_CONFIG.keys())}")

y_label, plot_fn = MODE_CONFIG[PLOT_MODE]                # get y-axis label and plotting function

# generate both plots (resistance and dwell time) in the chosen mode
plot_fn("resistance_MOhm", "R (MOhm)",        y_label, f"{PLOT_MODE}_resistance.html",  PLOT_MODE)
plot_fn("dwell_time_ms",   "Dwell Time (ms)", y_label, f"{PLOT_MODE}_dwell_time.html",  PLOT_MODE)