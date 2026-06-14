"""
Unified Grid Histogram + KDE Comparison
-----------------------------------------
Generates an interactive grid of plots:
    - Rows:    one per group + one comparison row at the bottom
    - Columns: one per variable (resistance, dwell time, EC)

Group rows: bar histogram with spline curve overlay (count on y-axis)
Comparison row: KDE density curves for configured groups overlaid

All subplots in the same column share the same x-axis range.

Requirements:
    pip install pandas plotly scipy numpy
"""

import pandas as pd                          # for loading and combining CSV metadata
import numpy as np                           # for numerical operations
import plotly.graph_objects as go            # for building interactive plots
from plotly.subplots import make_subplots    # for creating subplot grids
import plotly.io as pio                      # for saving interactive HTML files
from pathlib import Path                     # for navigating the folder structure
from scipy.stats import gaussian_kde         # for computing KDE curves
from scipy.interpolate import make_interp_spline  # for spline through bar tops

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIRS = [
    Path("modified_csvs/data_with_real_current"),       # directory with real current data
    Path("modified_csvs/data_with_recovered_current"),  # directory with recovered current data
]

COMPARISON_GROUPS = ["aLA_holo", "aLA_apo", "BSA"]  # groups to show in comparison row

# Automatically determine the optimal bin width
def auto_bin_width(data):
    """Compute bin width using Scott's rule: 3.5 * std / n^(1/3)"""
    return 3.5 * np.std(data) / (len(data) ** (1/3))  # Scott's rule for optimal bin width

X_CUTOFF_PERCENTILE = 95   # show only up to this percentile of data (e.g. 99 = cut top 1%)
KDE_POINTS = 500  # number of points to evaluate curves at

GROUPS = {
    "aLA_holo": "holo α-LA",  # group display names keyed by folder prefix
    "BSA":      "BSA",
    "aLA_apo":  "apo α-LA",
}

COLORS = {
    "aLA_holo": "#e6194b",  # red
    "BSA":      "#2698ba",  # cyan
    "aLA_apo":  "#33c110",  # green
}

VARIABLES = [
    ("resistance_MOhm", "R (MOhm)"),        # column 1: resistance
    ("dwell_time_ms",   "Dwell Time (ms)"), # column 2: dwell time
    ("area_nA_ms",      "EC (nA x ms)"),      # column 3: event charge
    ("dI_nA_max",       "dI max (nA)"),     # column 4: maximum deviation of the current
]

OUTPUT_FILE = Path("histograms") / "grid_histogram_kde.html"  # output file path

# ── Load and group CSVs ────────────────────────────────────────────────────────

group_data = {g: [] for g in GROUPS}  # dictionary to collect dataframes per group

for data_dir in DATA_DIRS:                                # loop over both data directories
    for csv_path in data_dir.rglob("*.csv"):              # recursively find all CSV files
        folder_name = csv_path.parent.name                # get the subfolder name
        for prefix in GROUPS:                             # check which group this folder belongs to
            if folder_name.startswith(prefix):            # match folder name to group prefix
                df = pd.read_csv(csv_path)                # load the CSV
                df["group"] = prefix                      # tag each row with its group name
                group_data[prefix].append(df)             # add to the group's list
                break                                     # stop checking prefixes once matched

group_dfs = {}
for prefix, dfs in group_data.items():                    # loop over each group
    if dfs:                                               # only process groups that have data
        group_dfs[prefix] = pd.concat(dfs, ignore_index=True)  # combine all CSVs for this group
        print(f"Group '{prefix}': {len(group_dfs[prefix])} total events")
    else:
        print(f"Group '{prefix}': no data found")         # warn if a group has no data

# ── Helper functions ───────────────────────────────────────────────────────────

def hex_to_rgba(hex_color, alpha=0.3):
    """Convert hex color string to rgba string with given alpha."""
    r = int(hex_color[1:3], 16)  # extract red component
    g = int(hex_color[3:5], 16)  # extract green component
    b = int(hex_color[5:7], 16)  # extract blue component
    return f"rgba({r},{g},{b},{alpha})"  # return rgba string

def spline_over_histogram(data, x_range, bin_width, n_points):
    """Fit a smooth spline through histogram bar tops."""
    bins        = np.arange(x_range[0], x_range[1] + bin_width, bin_width)  # bin edges matching histogram
    counts, edges = np.histogram(data, bins=bins)                            # compute bar counts
    centres     = (edges[:-1] + edges[1:]) / 2                              # bin centre x positions
    spline      = make_interp_spline(centres, counts, k=3)                   # fit cubic spline through bar tops
    x_vals      = np.linspace(centres[0], centres[-1], n_points)            # evenly spaced x points
    y_vals      = np.clip(spline(x_vals), 0, None)                          # evaluate spline, clip negatives to 0
    return x_vals, y_vals

# ── Compute shared x-axis ranges per column (variable) ────────────────────────

x_ranges = {}
for col_name, _ in VARIABLES:                             # loop over each variable
    all_vals    = pd.concat([df[col_name].dropna() for df in group_dfs.values()])  # all values across groups
    padding     = (all_vals.max() - all_vals.min()) * 0.05  # 5% padding on each side
    x_ranges[col_name] = [all_vals.min() - padding, all_vals.max() + padding]  # shared x range for this column

# ── Build subplot grid ────────────────────────────────────────────────────────

group_list = list(group_dfs.keys())                       # ordered list of group prefixes
n_rows     = len(group_list) + 1                          # one row per group + comparison row
n_cols     = len(VARIABLES)                               # one column per variable
row_titles = [GROUPS[p] for p in group_list] + ["Comparison"]  # row label for each row

# build subplot titles: variable names on first row only
subplot_titles = []
for row_idx in range(n_rows):                             # loop over rows
    for col_idx, (_, x_label) in enumerate(VARIABLES):   # loop over columns
        subplot_titles.append(x_label if row_idx == 0 else "")  # only label first row

fig = make_subplots(
    rows=n_rows, cols=n_cols,                             # grid dimensions
    subplot_titles=subplot_titles,                        # variable names as column headers
    horizontal_spacing=0.08,                              # spacing between columns
    vertical_spacing=0.10,                                # spacing between rows
)

# ── Fill group rows (bars + spline) ───────────────────────────────────────────

for row_idx, prefix in enumerate(group_list):             # loop over group rows
    for col_idx, (col_name, x_label) in enumerate(VARIABLES):  # loop over columns
        x_range = x_ranges[col_name]                      # shared x range for this column
        df      = group_dfs[prefix]                       # dataframe for this group
        data    = df[col_name].dropna().values            # data values, NaN removed
        bin_width = auto_bin_width(data)                  # compute optimal bin width for this variable and group
        color   = COLORS[prefix]                          # group color

        # bar histogram
        fig.add_trace(
            go.Histogram(
                x=data,                                   # data to bin
                xbins=dict(size=bin_width),               # automatic bin width
                marker_color=hex_to_rgba(color, 0.5),     # semi-transparent fill
                marker_line=dict(color=color, width=1),   # solid outline
                name=GROUPS[prefix],                      # name for hover
                showlegend=False,                         # no legend for group rows
            ),
            row=row_idx + 1, col=col_idx + 1,             # correct subplot cell
        )

        # spline curve through bar tops
        x_spl, y_spl = spline_over_histogram(data, x_range, bin_width, KDE_POINTS)
        fig.add_trace(
            go.Scatter(
                x=x_spl, y=y_spl,                         # spline curve
                mode="lines",                             # line only, no markers
                line=dict(color=color, width=2),          # group color, solid line
                name=GROUPS[prefix],                      # name for hover
                showlegend=False,                         # no legend for group rows
            ),
            row=row_idx + 1, col=col_idx + 1,             # correct subplot cell
        )

        data_max = np.percentile(data, X_CUTOFF_PERCENTILE) * 1.05         # cut off top outliers using percentile
        data_min = data.min() - (data.max() - data.min()) * 0.02            # small padding on the left
        fig.update_xaxes(range=[data_min, data_max], title_text=x_label,
                 title_font=dict(size=13), row=row_idx + 1, col=col_idx + 1)
        fig.update_yaxes(title_text="Count" if col_idx == 0 else "",
                         row=row_idx + 1, col=col_idx + 1)  # count label on first column only

# ── Fill comparison row (KDE density curves) ──────────────────────────────────

for col_idx, (col_name, x_label) in enumerate(VARIABLES):  # loop over columns
    x_range = x_ranges[col_name]                            # shared x range for this column

    for prefix in COMPARISON_GROUPS:                        # loop over comparison groups
        if prefix not in group_dfs:                         # skip missing groups
            print(f"  Warning: '{prefix}' not found — skipping.")
            continue

        data  = group_dfs[prefix][col_name].dropna().values  # data values, NaN removed
        color = COLORS[prefix]                               # group color

        kde   = gaussian_kde(data)                           # fit KDE to data
        x_kde = np.linspace(data.min(), data.max(), KDE_POINTS)  # x points over data range
        y_kde = kde(x_kde)                                   # evaluate density at each point

        fig.add_trace(
            go.Scatter(
                x=x_kde, y=y_kde,                            # KDE density curve
                mode="lines",                                # line only
                line=dict(color=color, width=2),             # group color
                fill="tozeroy",                              # fill area under curve
                fillcolor=hex_to_rgba(color, 0.15),          # semi-transparent fill
                name=GROUPS[prefix],                         # group name for legend
                showlegend=col_idx == 0,                     # show legend only in first column
            ),
            row=n_rows, col=col_idx + 1,                     # last row, current column
        )

    fig.update_xaxes(range=x_range, title_text=x_label,
                     title_font=dict(size=13), row=n_rows, col=col_idx + 1)  # x range and label
    fig.update_yaxes(title_text="Density" if col_idx == 0 else "",
                     row=n_rows, col=col_idx + 1)            # density label on first column only

# ── Add row labels on the left side ───────────────────────────────────────────

# compute y positions for each row centre in paper coordinates
row_height = 1.0 / n_rows                                # fractional height of each row

for row_idx, row_title in enumerate(row_titles):
    axis_idx = row_idx * n_cols + 1                                     # index of first subplot in this row
    yref_str = f"y{axis_idx} domain" if axis_idx > 1 else "y domain"   # y1 is just "y" in Plotly
    fig.add_annotation(
        text=f"<b>{row_title}</b>",
        xref="paper", 
        yref=yref_str,
        x=-0.07,                                                    # position to the left
        y=0.5,                                                      # always centred within the subplot domain
        showarrow=False,
        textangle=-90,
        font=dict(size=15, color="black"),
        xanchor="center",
        yanchor="middle",
    )

# ── Layout and save ───────────────────────────────────────────────────────────

fig.update_layout(
    height=300 * n_rows,                                     # scale height to number of rows
    width=400 * n_cols,                                      # scale width to number of columns
    barmode="overlay",                                       # overlay bars if needed
    legend_title="Comparison groups",                        # legend title
    margin=dict(l=120, r=20, t=60, b=40),                    # more left margin for labels
)

OUTPUT_FILE.parent.mkdir(exist_ok=True)                      # create output directory if needed
pio.write_html(fig, str(OUTPUT_FILE))                        # save interactive HTML file
fig.show()                                                   # open in browser
print(f"Saved: {OUTPUT_FILE}")                               # confirm output path