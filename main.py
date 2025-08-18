import wfdb
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import matplotlib.pyplot as plt
from wfdb import processing

# -------------------------------------------------------------------
# 1.  Load 12-lead ECG  (change path to your PTB-XL file)
# -------------------------------------------------------------------
# This path is relative to the dataset's root. Make sure you have the dataset
# downloaded and the path is correct.
rec_path = ("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
            "records500/00000/00001_hr")
rec = wfdb.rdrecord(rec_path)  # WFDB object
x = rec.p_signal  # (samples, 12/15)
fs = int(rec.fs)
lead_nms = rec.sig_name[:12]  # keep standard 12


# -------------------------------------------------------------------
# 2.  Helper: comprehensive denoising for every lead
# -------------------------------------------------------------------
def denoise_all_leads(ecg, fs, leads=None):
    """
    Applies a comprehensive denoising pipeline to each lead of an ECG signal.

    Args:
        ecg (np.ndarray): The raw ECG signal array of shape (samples, leads).
        fs (int): The sampling frequency in Hz.
        leads (list, optional): A list of lead names.

    Returns:
        tuple: A tuple containing the denoised signal and processing info.
    """
    n, L = ecg.shape
    nyq = fs / 2
    clean = np.zeros_like(ecg, dtype=float)

    for k in range(L):
        # ensure float array
        s = np.asarray(ecg[:, k], dtype=float)

        # --- (1) baseline wander ------------------------------------------------
        hp_cut = 0.5  # Hz
        b, a = butter(4, hp_cut / nyq, "high")
        s_hp = filtfilt(b, a, s)

        coeffs = pywt.wavedec(s, "db6", level=8)
        # <-- FIX: replace int 0 with zero-array of same shape
        coeffs[0] = np.zeros_like(coeffs[0])
        s_bw = pywt.waverec(coeffs, "db6")[:len(s)]

        s1 = 0.5 * (s_hp + s_bw)  # fusion

        # --- (2) power-line interference ---------------------------------------
        for f in (50, 100, 150):  # change to 60-Hz region if needed
            if f < nyq:
                b, a = iirnotch(f, Q=35, fs=fs)
                s1 = filtfilt(b, a, s1)

        c = pywt.wavedec(s1, "db4", level=6)
        for lvl in (2, 3):  # bands ≈ 31-125 Hz
            if lvl < len(c) and isinstance(c[lvl], np.ndarray):
                thr = 0.1 * np.std(c[lvl])
                c[lvl] = pywt.threshold(c[lvl], thr, "soft")
        s2 = pywt.waverec(c, "db4")[:len(s1)]

        # --- (3) high-frequency / EMG noise ------------------------------------
        d = pywt.wavedec(s2, "db4", level=6)
        sigma = np.median(np.abs(d[-1])) / 0.6745
        lam0 = sigma * np.sqrt(2 * np.log(len(s2)))
        dclean = [d[0]]
        for i in range(1, len(d)):
            lam = lam0 * (0.5 + 0.5 * i / 6)  # level-adaptive
            dclean.append(pywt.threshold(d[i], lam, "soft"))
        s3 = pywt.waverec(dclean, "db4")[:len(s2)]

        # --- (4) light smoothing -----------------------------------------------
        # choose window <= 15 and odd; require >=3 for savgol
        win = min(15, len(s3))
        if win % 2 == 0:
            win -= 1
        if win < 3:
            # too short to apply Savitzky-Golay — skip smoothing
            s4 = s3
        else:
            s4 = savgol_filter(s3, window_length=win, polyorder=3)

        clean[:, k] = s4

    info = dict(processing=[
        "High-pass + wavelet baseline removal",
        "50/60 Hz notch + wavelet detail suppression",
        "Wavelet soft-threshold (EMG/white noise)",
        "Savitzky-Golay final smoothing"
    ])
    return clean, info


# -------------------------------------------------------------------
# 3.  Helper: PQRST complex detection for a single lead
# -------------------------------------------------------------------
def detect_pqrst(signal, fs):
    """
    Detects P, Q, R, S, T peaks in a single ECG signal.

    Args:
        signal (np.ndarray): The 1D ECG signal.
        fs (int): The sampling frequency.

    Returns:
        dict: A dictionary with arrays of indices for each peak type.
    """
    # Use wfdb's XQRS for robust R-peak detection
    xqrs = processing.XQRS(sig=signal, fs=fs)
    xqrs.detect()
    r_locs = xqrs.qrs_inds

    # Define search windows based on typical durations
    qrs_win = int(0.06 * fs)  # QRS complex duration
    p_win = int(0.14 * fs)  # P-wave duration
    t_win = int(0.35 * fs)  # T-wave duration

    results = {'P': [], 'Q': [], 'R': [], 'S': [], 'T': []}

    for r in r_locs:
        # Find Q-peak (minimum before R)
        q_start = max(r - qrs_win, 0)
        q_idx = np.argmin(signal[q_start:r]) + q_start if q_start < r else r

        # Find S-peak (minimum after R)
        s_end = min(r + qrs_win, len(signal))
        s_idx = np.argmin(signal[r:s_end]) + r if r < s_end else r

        # Find P-peak (maximum before Q)
        p_start = max(q_idx - p_win, 0)
        p_idx = np.argmax(signal[p_start:q_idx]) + p_start if p_start < q_idx else q_idx

        # Find T-peak (maximum after S)
        t_end = min(s_idx + t_win, len(signal))
        t_idx = np.argmax(signal[s_idx:t_end]) + s_idx if s_idx < t_end else s_idx

        results['P'].append(p_idx)
        results['Q'].append(q_idx)
        results['R'].append(r)
        results['S'].append(s_idx)
        results['T'].append(t_idx)

    return results


# -------------------------------------------------------------------
# 4.  Run denoising and PQRST detection
# -------------------------------------------------------------------
# Denoise all leads
denoised, meta = denoise_all_leads(x[:, :12], fs, lead_nms)
print("Pipeline complete — 12 leads denoised.")

# Select a single lead for PQRST analysis (e.g., Lead II, which is typically lead index 1)
# or just the first lead
lead_idx = 0
denoised_lead = denoised[:, lead_idx]
raw_lead = x[:, lead_idx]
lead_name = lead_nms[lead_idx]

# Detect PQRST on the denoised signal
pqrst_results = detect_pqrst(denoised_lead, fs)
print(f"PQRST complexes detected for lead '{lead_name}'.")

# -------------------------------------------------------------------
# 5.  Quick visual check of the selected lead
# -------------------------------------------------------------------
sec = 10  # Visualize first 10 seconds
N = min(sec * fs, len(raw_lead))
t = np.arange(N) / fs

plt.figure(figsize=(14, 8))

# Plot raw and denoised signal
plt.plot(t, raw_lead[:N], color="gray", lw=.6, label="Raw")
plt.plot(t, denoised_lead[:N], color="blue", lw=1.5, label="Denoised")

# Plot the detected P, Q, R, S, T points
# We plot the points on the denoised signal's y-values
plt.plot(t[pqrst_results['P']], denoised_lead[pqrst_results['P']], 'o', color='purple', markersize=6, label='P Peak')
plt.plot(t[pqrst_results['Q']], denoised_lead[pqrst_results['Q']], 'o', color='green', markersize=6, label='Q Peak')
plt.plot(t[pqrst_results['R']], denoised_lead[pqrst_results['R']], 'o', color='red', markersize=6, label='R Peak')
plt.plot(t[pqrst_results['S']], denoised_lead[pqrst_results['S']], 'o', color='orange', markersize=6, label='S Peak')
plt.plot(t[pqrst_results['T']], denoised_lead[pqrst_results['T']], 'o', color='cyan', markersize=6, label='T Peak')

# Final plot settings
plt.title(f"ECG Signal for Lead '{lead_name}' — Denoised with PQRST Complexes")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

