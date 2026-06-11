"""
Calculate Event Statistics
---------------------------
Loads event traces from HDF5 files and their corresponding CSVs,
calculates post-event baseline, dI_nA_max and dI_nA_min for each
event, and saves updated CSVs to a separate directory.

New columns added:
    post_event_baseline_nA  : mean current in the post-event buffer
    dI_nA_max               : maximum current deviation from pre-event baseline
    dI_nA_min               : minimum current deviation from pre-event baseline

Requirements:
    pip install h5py numpy pandas
"""

import h5py                    # for reading HDF5 files
import numpy as np             # for numerical array operations
import pandas as pd            # for loading and saving CSV metadata
from pathlib import Path       # for navigating the folder structure

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIRS = [
    Path("./data/data_for_plots/data_with_real_current"),       # directory with real current data
    Path("./data/data_for_plots/data_with_recovered_current"),  # directory with recovered current data
]

OUTPUT_DIR        = Path("modified_csvs")  # directory to save updated CSVs
HDF5_GROUP        = "events"              # HDF5 group containing the event datasets
SAMPLING_RATE_KHZ = 50                    # sampling rate in kHz (50 samples per ms)

# ── Helper functions ───────────────────────────────────────────────────────────

def compute_baseline(region: np.ndarray) -> float:
    """Compute baseline as mean of region, skipping first 20% to avoid transients."""
    skip = max(1, len(region) // 5)      # skip first ~20% of region
    return region[skip:].mean()          # mean of remaining samples

# ── Process all folders ────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)  # create output directory if it doesn't exist

for data_dir in DATA_DIRS:                                    # loop over both data directories
    for csv_path in data_dir.rglob("*.csv"):                  # recursively find all CSV files
        h5_path = csv_path.with_suffix(".h5")                 # find matching HDF5 file

        if not h5_path.exists():                              # check if HDF5 file exists
            print(f"  Warning: no HDF5 found for {csv_path.name} — skipping.")
            continue                                          # skip if no matching HDF5

        meta = pd.read_csv(csv_path, index_col="event_name")  # load CSV with event_name as row index
        print(f"Processing {csv_path.name} ({len(meta)} events)...")

        with h5py.File(h5_path, "r") as f:                    # open HDF5 file for reading
            grp = f[HDF5_GROUP]                               # access the group containing event datasets

            for event_name, row in meta.iterrows():           # loop over each event in the CSV

                if event_name not in grp:                     # check if event exists in HDF5
                    print(f"  Warning: '{event_name}' not found in HDF5 — skipping.")
                    continue                                  # skip missing events

                trace         = np.array(grp[event_name], dtype=np.float64)  # load full trace
                start         = int(row["start"])             # event start index in samples
                end           = int(row["end"])               # event end index in samples
                pre_region   = trace[:start]                  # slice out the pre-event buffer region
                skip         = max(1, len(pre_region) // 5)   # skip first ~20% to avoid transients
                pre_baseline = pre_region[skip:].mean()       # compute mean of remaining pre-event samples as baseline

                # post-event buffer region
                post_region   = trace[end:]                   # slice out post-event buffer
                if len(post_region) > 1:                      # only compute if post-event buffer exists
                    post_baseline = compute_baseline(post_region)  # compute post-event baseline
                else:
                    post_baseline = np.nan                    # set to NaN if no post-event buffer

                # event region relative to pre-event baseline
                event_trace   = trace[start:end] - pre_baseline  # subtract pre-event baseline from event

                dI_max = event_trace.max()                    # maximum current deviation from baseline
                dI_min = event_trace.min()                    # minimum current deviation from baseline

                meta.at[event_name, "post_event_baseline_nA"] = post_baseline  # save post-event baseline
                meta.at[event_name, "dI_nA_max"]              = dI_max         # save max current deviation
                meta.at[event_name, "dI_nA_min"]              = dI_min         # save min current deviation

        # mirror the subfolder structure in the output directory
        rel_path   = csv_path.relative_to(data_dir.parent)   # path relative to data_for_plots
        out_path   = OUTPUT_DIR / rel_path                    # full output path
        out_path.parent.mkdir(parents=True, exist_ok=True)    # create output subfolder if needed
        meta.to_csv(out_path)                                 # save updated CSV to output directory
        print(f"  Saved → {out_path}")                        # confirm output path

print(f"\nDone! Updated CSVs saved to {OUTPUT_DIR}")          # confirm completion