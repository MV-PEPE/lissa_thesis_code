"""
DTW Analysis on Trimmed Events
--------------------------------
Loads trimmed event traces from an HDF5 file, subtracts the per-event
baseline and normalises amplitude so the peak dip = -1, then DTW-aligns
all events to a chosen reference event. Plots the aligned traces as an
overlay together with the mean trace, rolling std, and the reference event.

Requirements:
    pip install h5py numpy pandas dtaidistance matplotlib
"""

import h5py
import pandas as pd
import numpy as np
from dtaidistance import dtw
import matplotlib.pyplot as plt

H5_PATH  = "./data_trimmed/data302-0.5gain_doublepeak_forDTW_trimmed.h5"
CSV_PATH = "./data_trimmed/data302-0.5gain_doublepeak_forDTW_trimmed.csv"

DOWNSAMPLE_TO = 500  # samples per event (increase later if needed)

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

# ── 3. Normalize to baseline ───────────────────────────────────────────────────
# def normalize_to_baseline(signal, baseline_frac=0.1):
#     n = max(1, int(len(signal) * baseline_frac))
#     baseline = signal[:n].mean()
#     return signal - baseline

# ── 4. Pick reference event ─────────────────────────────────────────────────────
common_keys = [k for k in signals if k in meta.index]
# ref_key      = "event_00176"
mean_dwell = meta.loc[common_keys, "dwell_time_ms"].mean()
ref_key    = (meta.loc[common_keys, "dwell_time_ms"] - mean_dwell).abs().idxmin()  # pick event closest to mean dwell time

ref_signal = (signals[ref_key] - meta.at[ref_key, "baseline_nA"]) / meta.at[ref_key, "delta_I_baseline_nA"]  # baseline-subtract and normalise reference amplitude
ref_dwell    = meta.loc[ref_key, "dwell_time_ms"]
ref_trim_len  = meta.at[ref_key, "end"] - meta.at[ref_key, "start"]  # full trimmed trace length in samples (from updated start/end in CSV)
ref_total_ms  = ref_trim_len / SAMPLING_RATE_KHZ                      # convert full trimmed length to milliseconds
ref_time      = np.linspace(0, ref_total_ms, DOWNSAMPLE_TO)           # time axis covering the full trimmed trace including buffers

print(f"Reference: {ref_key}, dwell_time_ms: {ref_dwell:.2f}")

print(f"Reference: {ref_key} ({ref_dwell:.1f} ms) — aligning {len(common_keys)} events...")

# ── 5. DTW-align all events ────────────────────────────────────────────────────
aligned = []
# print(meta.loc[common_keys, "delta_I_baseline_nA"].describe())
print(meta.loc[common_keys, "dwell_time_ms"].describe())
for i, key in enumerate(common_keys):
    # print(f"{key}: delta_I_baseline_nA = {meta.at[key, 'delta_I_baseline_nA']:.6f}")
    sig = (signals[key] - meta.at[key, "baseline_nA"]) / meta.at[key, "delta_I_baseline_nA"]  # baseline-subtract and normalise amplitude so peak dip = -1

    path        = dtw.warping_path(ref_signal, sig)
    ref_idx     = np.array([p[0] for p in path])
    evt_idx     = np.array([p[1] for p in path])
    interp      = np.interp(np.arange(len(ref_signal)), ref_idx, sig[evt_idx])
    aligned.append(interp)
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(common_keys)} done")

aligned = np.array(aligned)

# ── 6. Shift time so deepest peak = t=0 ───────────────────────────────────────
ref_peak_idx = np.argmin(ref_signal)
ref_peak_ms  = ref_time[ref_peak_idx]
time_aligned = ref_time - ref_peak_ms

# ── 7. Mean and rolling std ────────────────────────────────────────────────────
mean_trace  = aligned.mean(axis=0)
rolling_std = pd.Series(mean_trace).rolling(window=20, center=True).std().values

# ── 8. Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for trace in aligned:
    ax.plot(time_aligned, trace, color="steelblue", alpha=0.15, linewidth=0.5)

ax.plot(time_aligned, mean_trace, color="darkred", linewidth=1.5, label="Mean aligned trace")
ax.plot(time_aligned, rolling_std, color="orange", linewidth=1.5, label="Rolling std of mean")
ax.plot(time_aligned, ref_signal, color="black", linewidth=1, linestyle="--", label="Reference")

ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Normalised current (peak dip = -1)")
ax.set_title("DTW-Aligned Event Overlays")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("plots/dtw_aligned_overlay.png", dpi=150)
plt.show()
print("Saved: plots/dtw_aligned_overlay.png")