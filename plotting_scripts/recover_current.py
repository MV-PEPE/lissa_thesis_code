"""
Recover Real Current
--------------------
Reads raw traces from an HDF5 file, multiplies each trace by 2 to
recover the real current (recorded current = 0.5 * real current),
and saves the corrected traces to a new HDF5 file.

Requirements:
    pip install h5py numpy
"""

import h5py   # for reading and writing HDF5 files
import numpy as np  # for numerical array operations
import pandas as pd # for reading and writing CSV files
from pathlib import Path  # for navigating the folder structure

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = Path("./data/data_for_plots/data_with_halfed_current")    # root folder containing subfolders with HDF5 files
OUTPUT_DIR = Path("./data/data_for_plots/data_with_recovered_current")  # root folder where corrected files will be saved
HDF5_GROUP = "events"                            # HDF5 group containing the event datasets

# ── Recover real current and save ─────────────────────────────────────────────
for h5_path in INPUT_DIR.rglob("*.h5"):                                    # recursively find all HDF5 files in input folder
    out_path = OUTPUT_DIR / h5_path.relative_to(INPUT_DIR)                 # mirror the subfolder structure in output folder
    out_path.parent.mkdir(parents=True, exist_ok=True)                     # create output subfolder if it doesn't exist yet

    with h5py.File(h5_path, "r") as f_in, h5py.File(out_path, "w") as f_out:  # open input for reading, output for writing
        grp_in  = f_in[HDF5_GROUP]                # access the group containing raw event traces
        grp_out = f_out.create_group(HDF5_GROUP)  # create the same group structure in the output file

        for key, val in f_in.attrs.items():  # iterate over all root-level attributes of the input file
            f_out.attrs[key] = val           # copy each attribute to preserve file-level metadata

        for event_name in grp_in.keys():                                        # loop over every event dataset
            trace = np.array(grp_in[event_name], dtype=np.float64)              # load the raw trace as a float64 array
            trace = trace * 2                                                    # recover real current: recorded = 0.5 * real

            ds = grp_out.create_dataset(event_name, data=trace)                 # save corrected trace to output HDF5
            for key, val in grp_in[event_name].attrs.items():                   # iterate over all attributes of the original dataset
                ds.attrs[key] = val                                              # copy each attribute to preserve per-event metadata

        print(f"Processed {len(grp_in)} events → {out_path}")  # confirm how many events were corrected in this file

    # copy the CSV file as-is (no changes needed)
    csv_path = h5_path.with_suffix(".csv")                                      # find the matching CSV file
    if csv_path.exists():                                                        # check if it exists
        out_csv = OUTPUT_DIR / csv_path.relative_to(INPUT_DIR)                  # mirror path in output folder

        csv = pd.read_csv(csv_path)                          # load the CSV file
        csv["area_nA_ms"]  = csv["area_nA_ms"]  * 2         # recover real current: correct area column
        csv["delta_I_nA"]  = csv["delta_I_nA"]  * 2         # recover real current: correct delta_I column
        csv.to_csv(out_csv, index=False)                     # save corrected CSV to output folder
        print(f"Saved corrected CSV → {out_csv}")            # confirm CSV was saved

print(f"\nDone! Corrected data saved to {OUTPUT_DIR}")  # confirm completion