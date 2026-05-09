"""Preprocess fMRI .mat files into JRNGC-compatible _x.npy / _gc.npy format.

The JRNGC fMRI loader splits data into train/eval portions.
For our pipeline we need the full time series as _x.npy and ground truth graph as _gc.npy.
"""
import numpy as np
import scipy.io as scio
import os, sys
sys.path.insert(0, "/root/autodl-tmp/GUOJI/JRNGC")
os.chdir("/root/autodl-tmp/GUOJI/JRNGC")

for d in [15, 50]:
    print(f"\nProcessing fMRI d={d}...")
    path = f"data/fMRI_{d}.mat"
    data = scio.loadmat(path)
    group_tm = int(data['Ntimepoints'][0][0])  # Cast to int (avoids uint8 overflow)
    n_subjects = data['net'].shape[0]
    print(f"  {n_subjects} subjects, {group_tm} timepoints each")

    for subject in range(min(n_subjects, 5)):  # First 5 subjects as "seeds"
        ts = data['ts'][group_tm * subject: group_tm * (subject + 1)]
        ts = np.swapaxes(ts, 0, 1).astype(np.float32)  # (d, T)
        m = np.mean(ts, axis=1, keepdims=True)
        sd = np.std(ts, axis=1, keepdims=True)
        ts = (ts - m) / (sd + 1e-8)  # Standardize

        net = np.swapaxes(data['net'][subject], 0, 1)  # (d, d)
        gc = (net != 0).astype(np.int32)[:, :, np.newaxis]  # (d, d, 1)

        # Save
        outdir = f"data/fmri/num_nodes_{d}/subject_{subject}/seed_0"
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "_x.npy"), ts)
        np.save(os.path.join(outdir, "_gc.npy"), gc)
        print(f"  subject {subject}: x={ts.shape}, gc={gc.shape}, "
              f"edges={int(gc.sum())}, range=[{ts.min():.2f},{ts.max():.2f}]")

print("\nDone. fMRI data preprocessed.")
