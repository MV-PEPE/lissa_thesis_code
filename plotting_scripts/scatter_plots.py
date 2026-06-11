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
import plotly.express as px  # for interactive scatter plots
import plotly.io as pio      # for saving interactive HTML files
from pathlib import Path         # for navigating the folder structure

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIRS = [
    Path("./data/data_for_plots/data_with_real_current"),       # directory with real current data
    Path("./data/data_for_plots/data_with_recovered_current"),  # directory with recovered current data
]

# group prefixes and their display names
GROUPS = {
    "aLA_holo_1": "aLA holo 1",
    "aLA_holo_2": "aLA holo 2",
    "BSA_1":      "BSA 1",
    "BSA_2":      "BSA 2",
    "BSA_3":      "BSA 3",
    "aLA_apo_1":  "aLA apo 1",
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

# marker shapes cycling through subfolders within each group
MARKERS = ["o", "^", "s", "D", "x", "P"]  # circle, triangle, square, diamond, cross, plus

OUTPUT_DIR = Path("scatter_plots")  # directory to save the output figures

# ── Load and group CSVs ────────────────────────────────────────────────────────

group_data = {g: [] for g in GROUPS}  # dictionary to collect dataframes per group

for data_dir in DATA_DIRS:                               # loop over both data directories
    for csv_path in data_dir.rglob("*.csv"):             # recursively find all CSV files
        folder_name = csv_path.parent.name               # get the subfolder name (e.g. aLA_holo_1_1)

        for prefix in GROUPS:                            # check which group this folder belongs to
            if folder_name.startswith(prefix):           # match folder name to group prefix
                df = pd.read_csv(csv_path)               # load the CSV
                df["group"]     = prefix           # tag each row with its group name
                df["subfolder"] = folder_name      # tag each row with its specific subfolder name
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
        labels={x_col: x_label, y_col: y_label},                  # axis labels
        hover_data=["event_name", "dwell_time_ms", "area_nA_ms", "delta_I_rel"],  # show on hover
        opacity=0.6,                                               # transparency
    )

    fig.update_traces(marker=dict(size=6))                         # set marker size
    fig.update_layout(legend_title="Group / Subfolder")            # legend title

    out_path = OUTPUT_DIR / filename.replace(".png", ".html")      # save as HTML instead of PNG
    pio.write_html(fig, str(out_path))                             # save interactive HTML file
    fig.show()                                                     # open in browser
    print(f"Saved: {out_path}")                                    # confirm the file was saved

# ── Generate 4 scatter plots ───────────────────────────────────────────────────

make_scatter("dwell_time_ms", "area_nA_ms",   "Dwell Time (ms)", "EC (nA ms)",      "ec_vs_dwell_time.png")
make_scatter("dwell_time_ms", "delta_I_rel",  "Dwell Time (ms)", "Relative ΔI",     "delta_I_rel_vs_dwell_time.png")
make_scatter("area_nA_ms",    "delta_I_rel",  "EC (nA ms)",      "Relative ΔI",     "delta_I_rel_vs_ec.png")
make_scatter("resistance_MOhm", "dwell_time_ms", "R (MOhm)",     "Dwell Time (ms)", "dwell_time_vs_resistance.png")