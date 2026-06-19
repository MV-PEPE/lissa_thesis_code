"""
DTW Analysis on Trimmed Events
--------------------------------
Loads trimmed event traces from an HDF5 file, subtracts the per-event
baseline, then DTW-aligns all events to a chosen reference event.
Saves the results (aligned traces, time axes, reference signal, raw
signals) to an .npz file for separate plotting.

Requirements:
    pip install h5py numpy pandas dtaidistance matplotlib
"""

import h5py
import pandas as pd
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt

H5_PATH  = "./data/data_for_dtw/data_trimmed/data501_0.5gain_twostepsbeforedip_forDTW_trimmed.h5"
CSV_PATH = "./data/data_for_dtw/data_trimmed/data501_0.5gain_twostepsbeforedip_forDTW_trimmed.csv"

OUTPUT_NPZ = "dtw_plots/dtw_results.npz"  # path to save the DTW results

DOWNSAMPLE_TO = 1500  # samples per event (increase later if needed)

# Use for no downsampling
# with h5py.File(H5_PATH, "r") as f:
#     max_len = max(len(f[f"events/{k}"][()]) for k in f["events"].keys())  # find longest trace in samples
# DOWNSAMPLE_TO = max_len  # use full resolution (no downsampling)

SAMPLING_RATE_KHZ = 50

# ── 1. Load & downsample signals ───────────────────────────────────────────────
signals = {}
with h5py.File(H5_PATH, "r") as f:
    for key in f["events"].keys():
        raw = f[f"events/{key}"][()]
        signals[key] = np.interp(
            np.linspace(0, len(raw)-1, DOWNSAMPLE_TO),
            np.arange(len(raw)), raw
        )

# ── 2. Load metadata ───────────────────────────────────────────────────────────
meta = pd.read_csv(CSV_PATH).set_index("event_name")

# ── 3. Pick reference event ─────────────────────────────────────────────────────
common_keys = [k for k in signals if k in meta.index]
# ref_key      = "event_00176"
mean_dwell = meta.loc[common_keys, "dwell_time_ms"].mean()
ref_key    = (meta.loc[common_keys, "dwell_time_ms"] - mean_dwell).abs().idxmin()  # pick event closest to mean dwell time

ref_signal    = (signals[ref_key] - meta.at[ref_key, "baseline_nA"])     # baseline-subtract
ref_dwell     = meta.loc[ref_key, "dwell_time_ms"]
ref_trim_len  = meta.at[ref_key, "end"] - meta.at[ref_key, "start"]   # full trimmed trace length in samples (from updated start/end in CSV)
ref_total_ms  = ref_trim_len / SAMPLING_RATE_KHZ                      # convert full trimmed length to milliseconds
ref_time      = np.linspace(0, ref_total_ms, DOWNSAMPLE_TO)           # time axis covering the full trimmed trace including buffers

print(f"Reference: {ref_key}, dwell_time_ms: {ref_dwell:.2f}")

print(f"Reference: {ref_key} ({ref_dwell:.1f} ms) — aligning {len(common_keys)} events...")

# ── 4. DTW-align all events ────────────────────────────────────────────────────
aligned = []                    # DTW-aligned traces
raw_signals_list = []           # raw baseline-subtracted traces (for raw overlay plot)
raw_time_list    = []           # per-event time axes (for raw overlay plot)

for i, key in enumerate(common_keys):
    sig = (signals[key] - meta.at[key, "baseline_nA"]) # baseline-subtract

    path        = dtw.warping_path(ref_signal, sig)
    ref_idx     = np.array([p[0] for p in path])
    evt_idx     = np.array([p[1] for p in path])
    interp      = np.interp(np.arange(len(ref_signal)), ref_idx, sig[evt_idx])
    aligned.append(interp)

    trace_ms  = (meta.at[key, "end"] - meta.at[key, "start"]) / SAMPLING_RATE_KHZ   # total trace length in ms
    time_axis = np.linspace(0, trace_ms, DOWNSAMPLE_TO)                             # this event's own time axis
    raw_signals_list.append(sig)
    raw_time_list.append(time_axis)

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(common_keys)} done")

aligned = np.array(aligned)
raw_signals = np.array(raw_signals_list)
raw_times   = np.array(raw_time_list)

# ── 5. (Optional) Shift time so deepest peak = t=0 ─────────────────────────────
# ref_peak_idx = np.argmin(ref_signal)
# ref_peak_ms  = ref_time[ref_peak_idx]
# time_aligned = ref_time - ref_peak_ms

time_aligned = ref_time  # t=0 stays at the start of the trimmed trace, no shift

# ── 6. Mean and rolling std ────────────────────────────────────────────────────
mean_trace  = aligned.mean(axis=0)
rolling_std = pd.Series(mean_trace).rolling(window=20, center=True).std().values

# ── 7. Save results ────────────────────────────────────────────────────────────
np.savez(
    OUTPUT_NPZ,
    aligned=aligned,                # DTW-aligned traces, shape (n_events, DOWNSAMPLE_TO)
    time_aligned=time_aligned,      # time axis for the DTW overlay plot, shape (DOWNSAMPLE_TO,)
    ref_signal=ref_signal,          # reference trace, shape (DOWNSAMPLE_TO,)
    mean_trace=mean_trace,          # mean of aligned traces, shape (DOWNSAMPLE_TO,)
    rolling_std=rolling_std,        # rolling std of mean trace, shape (DOWNSAMPLE_TO,)
    raw_signals=raw_signals,        # raw baseline-subtracted traces, shape (n_events, DOWNSAMPLE_TO)
    raw_times=raw_times,            # per-event time axes, shape (n_events, DOWNSAMPLE_TO)
    ref_key=ref_key,                # reference event name
)
print(f"Saved DTW results to {OUTPUT_NPZ}")
