"""
Event Scatter Plots - Multi-Group
----------------------------------
Loads event metadata from all CSVs in data_with_real_current and
data_with_recovered_current, groups them by folder prefix, and
generates 4 separate scatter plots with one color per group.

Requirements:
    pip install pandas matplotlib
"""

import pandas as pd              # for loading and combining CSV metadata
import plotly.express as px              # for interactive scatter plots
import plotly.graph_objects as go        # for combining traces into a grid
from plotly.subplots import make_subplots  # for creating the 2x2 grid
import plotly.io as pio                  # for saving interactive HTML files
from pathlib import Path         # for navigating the folder structure

# ── Configuration ─────────────────────────────────────────────────────────────

COMBINED_PAGE = True  # set to True to also generate a single 2x2 combined HTML page

DATA_DIRS = [
    Path("./data/data_for_plots/data_with_real_current"),       # directory with real current data
    Path("./data/data_for_plots/data_with_recovered_current"),  # directory with recovered current data
]

# group prefixes and their display names
GROUPS = {
    "aLA_holo_1": "holo α-LA 1",
    "aLA_holo_2": "holo α-LA 2",
    "BSA_1":      "BSA 1",
    "BSA_2":      "BSA 2",
    "BSA_3":      "BSA 3",
    "aLA_apo_1":  "apo α-LA 1",
}

# high-contrast colors for each group
COLORS = {
    "aLA_holo_1": "#e6194b",  # red
    "aLA_holo_2": "#f46642",  # orange
    "BSA_1":      "#2698ba",  # cyan
    "BSA_2":      "#0037ff",  # blue
    "BSA_3":      "#140083",  # dark blue
    "aLA_apo_1":  "#33c110",  # green
}

# marker shapes cycling through subfolders within each group (plotly symbol names, not matplotlib codes)
SYMBOLS = ["circle", "triangle-up", "square", "diamond", "x", "cross"]  # circle, triangle, square, diamond, x, plus

OUTPUT_DIR = Path("scatter_plots")  # directory to save the output figures

# ── Load and group CSVs ────────────────────────────────────────────────────────

group_data = {g: [] for g in GROUPS}  # dictionary to collect dataframes per group

for data_dir in DATA_DIRS:                               # loop over both data directories
    for csv_path in data_dir.rglob("*.csv"):             # recursively find all CSV files
        folder_name = csv_path.parent.name               # get the subfolder name (e.g. aLA_holo_1_1)

        for prefix in GROUPS:                            # check which group this folder belongs to
            if folder_name.startswith(prefix):           # match folder name to group prefix
                df = pd.read_csv(csv_path)               # load the CSV
                df["group"]     = prefix                 # tag each row with its group name
                df["subfolder"] = folder_name            # tag each row with its specific subfolder name
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

symbol_map = {}                                            # maps subfolder name -> plotly symbol
subfolder_label_map = {}                                   # maps subfolder name -> legend label (group name + index)
subfolder_order = []                                        # subfolder names in the order we want them to appear in the legend
for prefix, df in group_dfs.items():                       # loop over each group
    subfolders = sorted(df["subfolder"].unique())          # this group's subfolders, in a consistent order
    for i, subfolder in enumerate(subfolders):              # index restarts at 0 for every group
        symbol_map[subfolder] = SYMBOLS[i % len(SYMBOLS)]   # assign symbols starting from "circle" again
        index = subfolder[len(prefix) + 1:]                 # part of the subfolder name after "prefix_", e.g. "1"
        subfolder_label_map[subfolder] = f"{GROUPS[prefix]}.{index}"  # e.g. "α-LA holo 1.1"
        subfolder_order.append(subfolder)                   # record this subfolder's position in the desired legend order

# ── Plotting helper ────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)  # create output directory if it doesn't exist

def make_scatter(x_col, y_col, x_label, y_label, filename):
    """Create an interactive scatter plot for all groups with one color per group."""
    frames = []                                                    # list to collect all group dataframes
    for prefix, df in group_dfs.items():                           # loop over each group
        df = df.copy()                                             # avoid modifying the original dataframe
        df["group_label"] = GROUPS[prefix]                         # add display name column for plotly
        frames.append(df)                                          # add to list

    combined = pd.concat(frames, ignore_index=True)                # combine all groups into one dataframe

    fig = px.scatter(
        combined,
        x=x_col,                                                   # x axis column
        y=y_col,                                                   # y axis column
        color="group_label",                                       # color by group
        symbol="subfolder",                                        # shape by subfolder
        color_discrete_map={GROUPS[p]: COLORS[p] for p in GROUPS}, # use our custom colors
        symbol_map=symbol_map,                                     # assign symbols per subfolder, restarting at "circle" for each group
        category_orders={"subfolder": subfolder_order},             # force the legend order to follow our group-by-group, numerically sorted ordering
        labels={x_col: x_label, y_col: y_label},                   # axis labels
        hover_data=["event_name", "dwell_time_ms", "area_nA_ms", "delta_I_rel"],  # show on hover
        opacity=0.6,                                               # transparency
    )

    fig.update_traces(marker=dict(size=6))                         # set marker size
    
    for trace in fig.data:                                         # loop over each trace
        if "," in trace.name:                                      # legend entries are "group, subfolder"
            subfolder = trace.name.split(",")[1].strip()            # original subfolder name, e.g. "aLA_holo_1_1"
            trace.name = subfolder_label_map[subfolder]             # replace with "group index", e.g. "α-LA holo 1 1"

    fig.update_layout(
        legend_title="Group",                                      # updated legend title
        font=dict(size=16),                                        # bigger text everywhere (axis labels, ticks, legend)
        title_font=dict(size=18),                                  # bigger plot title
        legend=dict(itemsizing="constant"),  # makes legend markers a consistent, slightly larger size independent of plot markers
    )

    out_path = OUTPUT_DIR / filename.replace(".png", ".html")      # save as HTML instead of PNG
    pio.write_html(fig, str(out_path))                             # save interactive HTML file
    fig.show()                                                     # open in browser
    print(f"Saved: {out_path}")                                    # confirm the file was saved

    return fig                                                     # return figure for use in combined page

# ── Generate 4 scatter plots ───────────────────────────────────────────────────

plot_specs = [
    ("dwell_time_ms",   "area_nA_ms",  "Dwell Time (ms)", "EC (pC)",         "ec_vs_dwell_time.png"),
    ("dwell_time_ms",   "delta_I_rel", "Dwell Time (ms)", "Relative ΔI",     "delta_I_rel_vs_dwell_time.png"),
    ("area_nA_ms",      "delta_I_rel", "EC (pC)",         "Relative ΔI",     "delta_I_rel_vs_ec.png"),
    ("resistance_MOhm", "dwell_time_ms", "R (MOhm)",      "Dwell Time (ms)", "dwell_time_vs_resistance.png"),
]

figs = []                                                          # collect figures for combined page
for x_col, y_col, x_label, y_label, filename in plot_specs:        # loop over each plot spec
    fig = make_scatter(x_col, y_col, x_label, y_label, filename)   # generate individual plot
    figs.append((fig, x_label, y_label))                           # store for combined page

# ── Combined 2x2 page ────────────────────────────────────────────────────────

if COMBINED_PAGE:                                                  # only build combined page if enabled
    combined_fig = make_subplots(
        rows=2, cols=2,                                            # 2x2 grid
        subplot_titles=[f"{y} vs {x}" for _, x, y in figs],        # title each subplot
        vertical_spacing=0.15,                                     # reduce gap between rows (default is ~0.3 for 2x2)
    )

    PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)"]                        # one label per subplot, in the same order as plot_specs

    for i, (fig, x_label, y_label) in enumerate(figs):             # loop over each individual figure
        row = i // 2 + 1                                           # row index (1-indexed)
        col = i % 2 + 1                                            # column index (1-indexed)

        for trace in fig.data:                                     # loop over each trace in the figure
            trace.showlegend = (i == 0)                            # show legend only for first subplot's traces
            combined_fig.add_trace(trace, row=row, col=col)        # add trace to the combined figure

        combined_fig.update_xaxes(title_text=x_label, row=row, col=col)  # x-axis label
        combined_fig.update_yaxes(title_text=y_label, row=row, col=col)  # y-axis label

        combined_fig.add_annotation(                                  # add the panel letter to this subplot's corner
            text=PANEL_LETTERS[i], xref="x domain", yref="y domain",   # position relative to this subplot's own plot area (0-1), not the whole figure
            x=1, y=1, xanchor="right", yanchor="top",                   # top-right corner of the plot area
            xshift=-8, yshift=-8,                                       # nudge it slightly inward so it isn't flush against the border
            showarrow=False, font=dict(size=20),                        # no arrow pointing anywhere, readable size
            row=row, col=col,                                           # which subplot this label belongs to
        )

    combined_fig.update_layout(
        height=1100, width=2000,
        legend_title="Group",                                         # updated legend title
        font=dict(size=20),                                           # bigger text everywhere
        legend=dict(itemsizing="constant"),  # makes legend markers a consistent, slightly larger size independent of plot markers
    )
    
    for annotation in combined_fig.layout.annotations:                # subplot titles are stored as annotations
        annotation.font.size = 24                                      # increase subplot title size
    
    combined_out = OUTPUT_DIR / "combined_scatter_2x2.html"        # output path for combined page
    pio.write_html(combined_fig, str(combined_out))                # save combined HTML file
    combined_fig.show()                                            # open in browser
    print(f"Saved: {combined_out}")                                # confirm the file was saved
