"""
Unified Grid Histogram + KDE Comparison
-----------------------------------------
Generates a 7x3 interactive grid of plots:
    - Columns 1-6: one per group, each with bars + KDE overlay
    - Column 7:    overlaid KDE curves for configured groups
    - Rows:        count vs. resistance, count vs. dwell time, count vs. EC

All subplots in the same row share the same x-axis range.

Requirements:
    pip install pandas plotly scipy numpy
"""

import pandas as pd                        # for loading and combining CSV metadata
import numpy as np                         # for numerical operations
import plotly.graph_objects as go          # for building interactive plots
from plotly.subplots import make_subplots  # for creating subplot grids
import plotly.io as pio                    # for saving interactive HTML files
from pathlib import Path                   # for navigating the folder structure
from scipy.stats import gaussian_kde       # for computing KDE curves

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIRS = [
    Path("data/data_for_plots/data_with_real_current"),       # directory with real current data
    Path("data/data_for_plots/data_with_recovered_current"),  # directory with recovered current data
]

# groups to show in the comparison column (last column)
COMPARISON_GROUPS = ["aLA_holo", "aLA_apo", "BSA"]  # configure which groups to compare

BIN_WIDTH  = 10    # bin width for bar histograms (in x-axis units)
KDE_POINTS = 500  # number of points to evaluate KDE curve at

GROUPS = {
    "aLA_holo": "aLA holo",  # group display names keyed by folder prefix
    "BSA":      "BSA",
    "aLA_apo":  "aLA apo",
}

COLORS = {
    "aLA_holo": "#e6194b",  # red
    "BSA":      "#3cb44b",  # green
    "aLA_apo":  "#911eb4",  # purple
}

# variables to plot, one per row
VARIABLES = [
    ("resistance_MOhm", "R (MOhm)"),      # row 1: resistance
    ("dwell_time_ms",   "Dwell Time (ms)"),  # row 2: dwell time
    ("area_nA_ms",      "EC (nA ms)"),    # row 3: event charge
]

OUTPUT_FILE = Path("histograms") / "grid_histogram_kde.html"  # output file path

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

# ── KDE helper ─────────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color string to rgba string with given alpha."""
    r = int(hex_color[1:3], 16)   # extract red component
    g = int(hex_color[3:5], 16)   # extract green component
    b = int(hex_color[5:7], 16)   # extract blue component
    return f"rgba({r},{g},{b},{alpha})"  # return rgba string

# ── Compute shared x-axis ranges per row ──────────────────────────────────────

x_ranges = {}
for col_name, _ in VARIABLES:                            # loop over each variable (row)
    all_vals = pd.concat([
        df[col_name].dropna() for df in group_dfs.values()
    ])                                                   # combine all groups' values for this variable
    padding      = (all_vals.max() - all_vals.min()) * 0.05  # add 5% padding on each side
    x_ranges[col_name] = [all_vals.min() - padding, all_vals.max() + padding]  # shared x range for this row

# ── Build subplot grid ────────────────────────────────────────────────────────

group_list    = list(group_dfs.keys())                   # ordered list of group prefixes
n_groups      = len(group_list)                          # number of groups
n_cols        = n_groups + 1                             # +1 for comparison column
n_rows        = len(VARIABLES)                           # one row per variable

col_titles    = [GROUPS[p] for p in group_list] + ["Comparison"]  # column titles
subplot_titles = []
for row_idx in range(n_rows):                            # build subplot titles row by row
    for col_title in col_titles:                         # one title per subplot
        subplot_titles.append(col_title if row_idx == 0 else "")  # only show titles in first row

fig = make_subplots(
    rows=n_rows, cols=n_cols,                            # grid dimensions
    subplot_titles=subplot_titles,                       # titles for each subplot
    shared_xaxes=False,                                  # x ranges set manually per row
    horizontal_spacing=0.04,                             # spacing between columns
    vertical_spacing=0.08,                               # spacing between rows
)

# ── Fill in subplots ──────────────────────────────────────────────────────────

for row_idx, (col_name, x_label) in enumerate(VARIABLES):   # loop over rows (variables)
    x_range = x_ranges[col_name]                             # shared x range for this row

    for col_idx, prefix in enumerate(group_list):            # loop over group columns
        df    = group_dfs[prefix]                            # get dataframe for this group
        data  = df[col_name].dropna().values                 # get data, drop NaN
        color = COLORS[prefix]                               # group color

        # bars
        fig.add_trace(
            go.Histogram(
                x=data,                                      # data to bin
                xbins=dict(size=BIN_WIDTH),                  # fixed bin width
                marker_color=hex_to_rgba(color, 0.5),        # semi-transparent fill
                marker_line=dict(color=color, width=1),      # solid outline
                showlegend=False,                            # no legend for individual subplots
                name=GROUPS[prefix],                         # name for hover
            ),
            row=row_idx + 1, col=col_idx + 1,               # place in correct subplot cell
        )

        # KDE curve overlaid on bars
        from scipy.interpolate import make_interp_spline                        # for smooth spline through bar tops
        counts, bin_edges = np.histogram(data, bins=np.arange(x_range[0], x_range[1] + BIN_WIDTH, BIN_WIDTH))  # compute histogram counts matching the bars
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2                     # compute bin centre x positions
        spline      = make_interp_spline(bin_centres, counts, k=3)              # fit cubic spline through bin centres and counts
        x_kde       = np.linspace(bin_centres[0], bin_centres[-1], KDE_POINTS)  # evenly spaced x points
        y_kde       = np.clip(spline(x_kde), 0, None)                          # evaluate spline, clip negatives to 0
        fig.add_trace(
            go.Scatter(
                x=x_kde, y=y_kde,                            # KDE curve
                mode="lines",                                # line only
                line=dict(color=color, width=2),             # group color, solid line
                showlegend=False,                            # no legend for individual subplots
                name=GROUPS[prefix],                         # name for hover
            ),
            row=row_idx + 1, col=col_idx + 1,               # place in correct subplot cell
        )

        # apply shared x range
        fig.update_xaxes(range=x_range, row=row_idx + 1, col=col_idx + 1)  # set x range for this cell
        fig.update_yaxes(title_text="Count" if col_idx == 0 else "",        # y label only on first column
                         row=row_idx + 1, col=col_idx + 1)
        fig.update_yaxes(title_text="Density", row=row_idx + 1, col=n_cols)  # density label for comparison column

    # ── Comparison column (last column) ───────────────────────────────────────

    for prefix in COMPARISON_GROUPS:                         # loop over configured comparison groups
        if prefix not in group_dfs:                          # skip if group has no data
            print(f"  Warning: comparison group '{prefix}' not found — skipping.")
            continue

        df    = group_dfs[prefix]                            # get dataframe for this group
        data  = df[col_name].dropna().values                 # get data, drop NaN
        color = COLORS[prefix]                               # group color

        kde    = gaussian_kde(data)
        x_kde  = np.linspace(data.min(), data.max(), KDE_POINTS)
        y_kde  = kde(x_kde)
        fig.add_trace(
            go.Scatter(
                x=x_kde, y=y_kde,                            # KDE curve
                mode="lines",                                # line only
                line=dict(color=color, width=2),             # group color
                fill="tozeroy",                              # fill area under curve
                fillcolor=hex_to_rgba(color, 0.15),          # semi-transparent fill
                name=GROUPS[prefix],                         # group name for legend
                showlegend=row_idx == 0,                     # only show legend entry in first row
            ),
            row=row_idx + 1, col=n_cols,                     # place in last column
        )

    # apply shared x range to comparison column
    fig.update_xaxes(range=x_range, row=row_idx + 1, col=n_cols)  # set x range for comparison column

    # label every row with its corresponding variable
    for col_idx in range(n_cols):
        fig.update_xaxes(title_text=x_label, row=row_idx + 1, col=col_idx + 1)

# ── Layout and save ───────────────────────────────────────────────────────────

fig.update_layout(
    height=350 * n_rows,                                     # scale height to number of rows
    width=450 * n_cols,                                      # scale width to number of columns
    legend_title="Comparison groups",                        # legend title
    barmode="overlay",                                       # overlay bars (for any overlapping histograms)
)

OUTPUT_FILE.parent.mkdir(exist_ok=True)                      # create output directory if needed
pio.write_html(fig, str(OUTPUT_FILE))                        # save interactive HTML file
fig.show()                                                   # open in browser
print(f"Saved: {OUTPUT_FILE}")                               # confirm output path