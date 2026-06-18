"""
Event Density Plots - Multi-Group
------------------------------------
Loads event metadata from all CSVs in data_with_real_current and
data_with_recovered_current, groups them by folder prefix, and
generates 4 separate 2D density plots: filled, colored density blobs,
one color per group, all groups overlaid on the same plot with some
transparency so overlapping regions blend. This is the density-plot
analog of scatter_plots.py.

Requirements:
    pip install pandas plotly
"""

import pandas as pd                          # for loading and combining CSV metadata
import plotly.express as px                  # for the density contour plots
import plotly.io as pio                       # for saving interactive HTML files
from plotly.subplots import make_subplots     # for creating the 2x2 grid
from pathlib import Path                      # for navigating the folder structure

# ── Configuration ─────────────────────────────────────────────────────────────

COMBINED_PAGE = True  # set to True to also generate a single 2x2 combined HTML page

DATA_DIRS = [
    Path("./data/data_for_plots/data_with_real_current"),       # directory with real current data
    Path("./data/data_for_plots/data_with_recovered_current"),  # directory with recovered current data
]

# group prefixes and their display names (same groups as scatter_plots.py)
GROUPS = {
    "aLA_holo": "holo α-LA",
    "BSA":      "BSA",
    "aLA_apo":  "apo α-LA",
}

# high-contrast colors for each group (same colors as scatter_plots.py, for consistency)
COLORS = {
    "aLA_holo": "#e6194b",  # red
    "BSA":      "#2698ba",  # cyan
    "aLA_apo":  "#33c110",  # green
}

LABEL_TO_COLOR = {GROUPS[p]: COLORS[p] for p in GROUPS}  # group display name -> hex color; used for the legend AND for hand-building each trace's fill colorscale below

HISTNORM = None #"probability density"  # normalize each group's density to integrate to 1, so group size doesn't bias the comparison (set to None for raw counts instead)
N_CONTOURS = 6                    # number of filled density bands per group (fewer = chunkier blobs, more = smoother gradient)
FILL_OPACITY = 0.65                # trace opacity, so overlapping groups' blobs blend instead of one fully covering the other

OUTPUT_DIR = Path("density_plots")  # directory to save the output figures

# ── Load and group CSVs ────────────────────────────────────────────────────────

group_data = {g: [] for g in GROUPS}  # dictionary to collect dataframes per group

for data_dir in DATA_DIRS:                               # loop over both data directories
    for csv_path in data_dir.rglob("*.csv"):             # recursively find all CSV files
        folder_name = csv_path.parent.name               # get the subfolder name (e.g. aLA_holo_1_1)

        for prefix in GROUPS:                            # check which group this folder belongs to
            if folder_name.startswith(prefix):           # match folder name to group prefix
                df = pd.read_csv(csv_path)               # load the CSV
                df["group"]     = prefix           # tag each row with its group name
                df["subfolder"] = folder_name      # tag each row with its specific subfolder name (not used here, kept for consistency with scatter_plots.py)
                group_data[prefix].append(df)            # add to the group's list
                print(f"  Loaded {len(df)} events from {csv_path.name} → group '{prefix}'")
                break                                    # stop checking prefixes once matched

# concatenate all dataframes within each group
group_dfs = {}
for prefix, dfs in group_data.items():                   # loop over each group
    if dfs:                                              # only process groups that have data
        group_dfs[prefix] = pd.concat(dfs, ignore_index=True)  # combine all CSVs for this group
        print(f"Group '{prefix}': {len(group_dfs[prefix])} total events")
    else:
        print(f"Group '{prefix}': no data found")        # warn if a group has no data

# ── Plotting helper ────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)  # create output directory if it doesn't exist

def make_density(x_col, y_col, x_label, y_label, filename):
    """Create an interactive 2D density plot: filled, colored blobs, one per group, overlaid with transparency."""
    frames = []                                                    # list to collect all group dataframes
    for prefix, df in group_dfs.items():                           # loop over each group
        df = df.copy()                                             # avoid modifying the original dataframe
        df["group_label"] = GROUPS[prefix]                         # add display name column for plotly
        frames.append(df)                                          # add to list

    combined = pd.concat(frames, ignore_index=True)                # combine all groups into one dataframe

    fig = px.density_contour(
        combined,
        x=x_col,                                                   # x axis column
        y=y_col,                                                   # y axis column
        color="group_label",                                       # one filled blob per group
        color_discrete_map=LABEL_TO_COLOR,                          # use our custom colors
        labels={x_col: x_label, y_col: y_label},                   # axis labels
        histnorm=HISTNORM,                                          # normalize density so group size doesn't bias the comparison
    )

    fig.update_traces(
        contours_coloring="fill",       # filled bands, not just outlines, so each group reads as a solid colored blob
        ncontours=N_CONTOURS,            # how many density bands to draw per group
        opacity=FILL_OPACITY,            # let overlapping groups' blobs blend together instead of fully covering each other
        showscale=False,                 # hide the per-trace colorbar (the legend already identifies each group)
    )

    for trace in fig.data:                                          # color_discrete_map only colors the outline, not the fill, so build each trace's own fill colorscale by hand
        color = LABEL_TO_COLOR[trace.name]                          # this trace's assigned group color
        trace.colorscale = [[0, "rgba(0,0,0,0)"], [1, color]]       # fully transparent at low density, solid group color at the peak

    fig.update_layout(
        legend_title="Group",                                         # legend title
        font=dict(size=16),                                           # bigger text everywhere (axis labels, ticks, legend)
        title_font=dict(size=18),                                     # bigger plot title
        legend=dict(itemsizing="constant"),  # makes legend markers a consistent size
    )

    out_path = OUTPUT_DIR / filename                                # save as HTML
    pio.write_html(fig, str(out_path))                             # save interactive HTML file
    fig.show()                                                     # open in browser
    print(f"Saved: {out_path}")                                    # confirm the file was saved

    return fig                                                     # return figure for use in combined page

# ── Generate 4 density plots ───────────────────────────────────────────────────

plot_specs = [
    ("dwell_time_ms",   "area_nA_ms",    "Dwell Time (ms)", "EC (pC)",         "ec_vs_dwell_time.html"),
    ("dwell_time_ms",   "delta_I_rel",   "Dwell Time (ms)", "Relative ΔI",     "delta_I_rel_vs_dwell_time.html"),
    ("area_nA_ms",      "delta_I_rel",   "EC (pC)",         "Relative ΔI",     "delta_I_rel_vs_ec.html"),
    ("resistance_MOhm", "dwell_time_ms", "R (MOhm)",        "Dwell Time (ms)", "dwell_time_vs_resistance.html"),
]

figs = []                                                          # collect figures for combined page
for x_col, y_col, x_label, y_label, filename in plot_specs:        # loop over each plot spec
    fig = make_density(x_col, y_col, x_label, y_label, filename)   # generate individual plot
    figs.append((fig, x_label, y_label))                           # store for combined page

# ── Combined 2x2 page ────────────────────────────────────────────────────────

if COMBINED_PAGE:                                                  # only build combined page if enabled
    combined_fig = make_subplots(
        rows=2, cols=2,                                            # 2x2 grid
        subplot_titles=[f"{y} vs {x}" for _, x, y in figs],        # title each subplot
        vertical_spacing=0.15,                                     # reduce gap between rows (default is ~0.3 for 2x2)
    )

    PANEL_LETTERS = ["a)", "b)", "c)", "d)"]                        # one label per subplot, in the same order as plot_specs

    for i, (fig, x_label, y_label) in enumerate(figs):             # loop over each individual figure
        row = i // 2 + 1                                           # row index (1-indexed)
        col = i % 2 + 1                                            # column index (1-indexed)

        for trace in fig.data:                                     # loop over each trace in the figure
            trace.showlegend = (i == 0)                            # show legend only for first subplot's traces
            combined_fig.add_trace(trace, row=row, col=col)        # add trace to the combined figure

        combined_fig.update_xaxes(title_text=x_label, row=row, col=col)  # x-axis label
        combined_fig.update_yaxes(title_text=y_label, row=row, col=col)  # y-axis label

        combined_fig.add_annotation(                                  # add the panel letter to this subplot's corner
            text=PANEL_LETTERS[i], xref="x domain", yref="y domain",   # position relative to this subplot's own plot area, not the whole figure
            x=1, y=1, xanchor="right", yanchor="top",                   # top-right corner of the plot area
            xshift=-8, yshift=-8,                                       # nudge it slightly inward so it isn't flush against the border
            showarrow=False, font=dict(size=20),                        # no arrow pointing anywhere, readable size
            row=row, col=col,                                           # which subplot this label belongs to
        )

    combined_fig.update_layout(
        height=1100, width=2000,
        legend_title="Group",                                          # legend title
        font=dict(size=20),                                           # bigger text everywhere
        legend=dict(itemsizing="constant"),  # makes legend markers a consistent size
    )

    for annotation in combined_fig.layout.annotations:                # subplot titles AND panel letters are stored as annotations
        annotation.font.size = 24                                      # increase subplot title size (also affects the panel letters, same as in scatter_plots.py)

    combined_out = OUTPUT_DIR / "combined_density_2x2.html"        # output path for combined page
    pio.write_html(combined_fig, str(combined_out))                # save combined HTML file
    combined_fig.show()                                            # open in browser
    print(f"Saved: {combined_out}")                                # confirm the file was saved