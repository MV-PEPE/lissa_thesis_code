"""
Trim Events with Buffer
-----------------------
Reads raw traces from an HDF5 file, trims each event to its window
(start/end from CSV) plus a buffer before and after, and saves the
trimmed traces to a new HDF5 file and an updated CSV.

Buffer sizes (in samples, at 50 kHz):
    before = dwell_time_samples / 3
    after  = dwell_time_samples * 2

Requirements:
    pip install h5py numpy pandas
"""

import h5py          # for reading and writing HDF5 files
import numpy as np   # for numerical array operations
import pandas as pd  # for reading and manipulating the CSV metadata

# ── Configuration ─────────────────────────────────────────────────────────────

HDF5_INPUT        = "./data/data302-0.5gain_doublepeak_forDTW.h5"                  # path to the original HDF5 file
HDF5_OUTPUT       = "./data_trimmed/data302-0.5gain_doublepeak_forDTW_trimmed.h5"  # path where the trimmed HDF5 will be saved
CSV_INPUT         = "./data/data302-0.5gain_doublepeak_forDTW.csv"                 # path to the original event metadata CSV
CSV_OUTPUT        = "./data_trimmed/data302-0.5gain_doublepeak_forDTW_trimmed.csv" # path where the updated CSV will be saved
HDF5_GROUP        = "events"                                                           # HDF5 group that contains the event datasets
SAMPLING_RATE_KHZ = 50  # acquisition sampling rate in kHz (50 kHz = 50 samples per ms)

# ── Load metadata ─────────────────────────────────────────────────────────────

meta = pd.read_csv(CSV_INPUT, index_col="event_name")  # load CSV with event_name as the row index
print(f"Loaded {len(meta)} events from CSV.")          # confirm how many events were found

# ── Trim and save ─────────────────────────────────────────────────────────────

skipped = []  # keep track of events that are missing from the HDF5 file

with h5py.File(HDF5_INPUT, "r") as f_in, h5py.File(HDF5_OUTPUT, "w") as f_out:  # open input HDF5 for reading, output for writing
    grp_in  = f_in[HDF5_GROUP]                # access the group containing raw event traces in the input file
    grp_out = f_out.create_group(HDF5_GROUP)  # create the same group structure in the output file

    for key, val in f_in.attrs.items():  # iterate over all root-level attributes of the input file (e.g. sampling rate)
        f_out.attrs[key] = val           # copy each attribute to the output file to preserve file-level metadata

    for event_name, row in meta.iterrows():  # loop over each event row in the CSV metadata

        if event_name not in grp_in:                                             # check if this event exists in the HDF5 file
            print(f"  Warning: '{event_name}' not found in HDF5 — skipping.")    # warn if it is missing
            skipped.append(event_name)                                           # record the missing event name
            continue                                                             # skip to the next event

        trace = np.array(grp_in[event_name], dtype=np.float64)  # load the full raw current trace as a float64 array

        dwell_samples = int(round(row["dwell_time_ms"] * SAMPLING_RATE_KHZ))  # convert dwell time from ms to samples (ms × 50 = samples)

        buf_before = int(round(dwell_samples / 3))  # buffer before event: one third of the dwell time in samples
        buf_after  = dwell_samples * 2              # buffer after event: twice the dwell time in samples

        start = int(row["start"])  # event start index in samples (from CSV)
        end   = int(row["end"])    # event end index in samples (from CSV)

        trim_start = max(0, start - buf_before)        # start of trimmed trace, clamped to 0 so it never goes negative
        trim_end   = min(len(trace), end + buf_after)  # end of trimmed trace, clamped to trace length so it never overflows

        trimmed = trace[trim_start:trim_end]  # slice the trace to the buffered event window
        
        pre_event = trace[trim_start : start]    # slice out the pre-event buffer region
        skip      = max(1, len(pre_event) // 5)  # skip first ~20% of buffer to avoid transients
        baseline  = pre_event[skip:].mean()      # compute mean of the remaining pre-event samples as baseline
        event_trace = trace[start:end]           # slice out just the event portion (no buffers)
        meta.at[event_name, "delta_I_baseline_nA"] = baseline - event_trace.min()  # max current drop relative to computed baseline

        ds = grp_out.create_dataset(event_name, data=trimmed)  # save the trimmed trace as a new dataset in the output HDF5
        for key, val in grp_in[event_name].attrs.items():      # iterate over all attributes of the original dataset
            ds.attrs[key] = val                                # copy each attribute to the new dataset to preserve per-event metadata

        meta.at[event_name, "start"] = trim_start      # overwrite start with the buffered trim start
        meta.at[event_name, "end"]   = trim_end        # overwrite end with the buffered trim end
        meta.at[event_name, "baseline_nA"] = baseline  # save the computed baseline current value to metadata

print(f"\nDone. {len(meta) - len(skipped)} events trimmed.")  # report how many events were successfully processed
if skipped:                                                    # only print skipped list if there are any
    print(f"Skipped {len(skipped)} events: {skipped}")        # list the names of skipped events

# ── Save updated CSV ──────────────────────────────────────────────────────────

meta_out = meta.drop(index=skipped)  # remove skipped events from the metadata before saving
meta_out.to_csv(CSV_OUTPUT)          # save the updated metadata (with new trim columns) to a CSV file
print(f"Saved trimmed HDF5 → {HDF5_OUTPUT}")  # confirm HDF5 output path
print(f"Saved updated CSV  → {CSV_OUTPUT}")   # confirm CSV output path

# ── Quick sanity check ────────────────────────────────────────────────────────

print("\nTrim length stats (samples):")              # header for the summary stats
# print(meta_out["trim_length"].describe().astype(int))  # print min/max/mean/std of trimmed trace lengths to spot outliers
