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
import matplotlib.pyplot as plt  # for plotting
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
    """Create a scatter plot for all groups with one color per group."""
    fig, ax = plt.subplots(figsize=(10, 6))              # create figure

    for prefix, df in group_dfs.items():                               # loop over each group
        for i, (subfolder, sub_df) in enumerate(df.groupby("subfolder")):  # loop over each subfolder within the group
            ax.scatter(
                sub_df[x_col], sub_df[y_col],                          # x and y data
                color=COLORS[prefix],                                   # color represents the group
                marker=MARKERS[i % len(MARKERS)],                       # shape represents the subfolder
                label=f"{GROUPS[prefix]} - {subfolder}",               # legend entry with group and subfolder name
                alpha=0.5,                                              # transparency
                s=15,                                                   # marker size
            )

    ax.set_xlabel(x_label)                               # x-axis label
    ax.set_ylabel(y_label)                               # y-axis label
    ax.legend(loc="upper right", title="Group")          # legend with group colors
    plt.tight_layout()                                   # prevent labels from being clipped
    out_path = OUTPUT_DIR / filename                     # full output path
    plt.savefig(out_path, dpi=150)                       # save figure to file
    plt.show()                                           # display figure interactively
    print(f"Saved: {out_path}")                          # confirm the file was saved
    plt.close()                                          # close figure to free memory

# ── Generate 4 scatter plots ───────────────────────────────────────────────────

make_scatter("dwell_time_ms", "area_nA_ms",   "Dwell Time (ms)", "EC (nA ms)",      "ec_vs_dwell_time.png")
make_scatter("dwell_time_ms", "delta_I_rel",  "Dwell Time (ms)", "Relative ΔI",     "delta_I_rel_vs_dwell_time.png")
make_scatter("area_nA_ms",    "delta_I_rel",  "EC (nA ms)",      "Relative ΔI",     "delta_I_rel_vs_ec.png")
make_scatter("resistance_MOhm", "dwell_time_ms", "R (MOhm)",     "Dwell Time (ms)", "dwell_time_vs_resistance.png")