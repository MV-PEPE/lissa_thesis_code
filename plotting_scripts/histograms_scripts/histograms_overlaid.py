"""
Overlaid Interactive Histograms
---------------------------------
Loads event metadata from all CSVs in data_with_real_current and
data_with_recovered_current, groups them by folder prefix, and
generates two interactive overlaid histograms (count vs. resistance
and count vs. dwell time) with one color per group.

Requirements:
    pip install pandas plotly
"""

import pandas as pd              # for loading and combining CSV metadata
import plotly.graph_objects as go  # for building interactive histograms
import plotly.io as pio          # for saving interactive HTML files
from pathlib import Path         # for navigating the folder structure

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIRS = [
    Path("data/data_for_plots/data_with_real_current"),       # directory with real current data
    Path("data/data_for_plots/data_with_recovered_current"),  # directory with recovered current data
]

for data_dir in DATA_DIRS:
    print(f"Looking in: {data_dir.resolve()}")  # print the absolute path being searched
    print(f"Exists: {data_dir.exists()}")        # check if the directory actually exists

GROUPS = {
    "aLA_holo_1": "aLA holo 1",  # group display names
    "BSA_1":      "BSA 1",
    "BSA_2":      "BSA 2",
    "BSA_3":      "BSA 3",
    "aLA_apo_1":  "aLA apo 1",
    "aLA_holo_2": "aLA holo 2",
}

COLORS = {
    "aLA_holo_1": "#e6194b",  # red
    "BSA_1":      "#3cb44b",  # green
    "BSA_2":      "#4363d8",  # blue
    "BSA_3":      "#f58231",  # orange
    "aLA_apo_1":  "#911eb4",  # purple
    "aLA_holo_2": "#42d4f4",  # cyan
}

OUTPUT_DIR = Path("histograms/histograms_overlaid")  # directory to save the output HTML files

# ── Load and group CSVs ────────────────────────────────────────────────────────

group_data = {g: [] for g in GROUPS}  # dictionary to collect dataframes per group

for data_dir in DATA_DIRS:                               # loop over both data directories
    for csv_path in data_dir.rglob("*.csv"):             # recursively find all CSV files
        folder_name = csv_path.parent.name               # get the subfolder name
        print(f"  Found folder: '{folder_name}'")  # debug: print folder name to check against group prefixes
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

# ── Plotting helper ────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)  # create output directory if it doesn't exist

def make_overlaid_histogram(col, x_label, filename):
    """Create an overlaid step histogram for all groups."""
    fig = go.Figure()                                    # create empty figure

    for prefix, df in group_dfs.items():                 # loop over each group
        fig.add_trace(go.Histogram(
            x=df[col],                                   # data column to plot
            name=GROUPS[prefix],                         # group display name for legend
            marker_color=COLORS[prefix],                 # group color
            histnorm="",                                 # use raw counts on y-axis
            opacity=1.0,                                 # fully opaque lines
            xbins=dict(size=10),  # fixed bin width of 2 MOhm for all groups
        ))

    fig.update_traces(histfunc="count")                  # count events per bin

    # step/line style: no fill, just the outline
    fig.update_traces(
        marker=dict(line=dict(width=2)),                 # outline width
        selector=dict(type="histogram"),                 # apply to all histograms
    )
    fig.update_layout(
        barmode="overlay",                               # overlay all groups on same plot
        bargap=0,                                        # no gap between bars
        xaxis_title=x_label,                            # x-axis label
        yaxis_title="Count",                             # y-axis label
        legend_title="Group",                           # legend title
    )

    out_path = OUTPUT_DIR / filename                     # full output path
    pio.write_html(fig, str(out_path))                   # save interactive HTML file
    fig.show()                                           # open in browser
    print(f"Saved: {out_path}")                          # confirm the file was saved

# ── Generate histograms ────────────────────────────────────────────────────────

make_overlaid_histogram("resistance_MOhm", "R (MOhm)",     "overlaid_histogram_resistance.html")
make_overlaid_histogram("dwell_time_ms",   "Dwell Time (ms)", "overlaid_histogram_dwell_time.html")