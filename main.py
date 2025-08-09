import wfdb
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1.  Load 12-lead ECG  (change path to your PTB-XL file)
# -------------------------------------------------------------------
rec_path = ("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/"
            "records500/00000/00001_hr")
rec      = wfdb.rdrecord(rec_path)              # WFDB object
x        = rec.p_signal                         # (samples, 12/15)
fs       = int(rec.fs)
lead_nms = rec.sig_name[:12]                    # keep standard 12

# -------------------------------------------------------------------
# 2.  Helper: comprehensive denoising for every lead
# -------------------------------------------------------------------
def denoise_all_leads(ecg, fs, leads=None):
    n, L        = ecg.shape
    nyq         = fs / 2
    clean       = np.zeros_like(ecg, dtype=float)

    for k in range(L):
        # ensure float array
        s = np.asarray(ecg[:, k], dtype=float)

        # --- (1) baseline wander ------------------------------------------------
        hp_cut    = 0.5                         # Hz
        b, a      = butter(4, hp_cut/nyq, "high")
        s_hp      = filtfilt(b, a, s)

        coeffs    = pywt.wavedec(s, "db6", level=8)
        # <-- FIX: replace int 0 with zero-array of same shape
        coeffs[0] = np.zeros_like(coeffs[0])
        s_bw      = pywt.waverec(coeffs, "db6")[:len(s)]

        s1        = 0.5 * (s_hp + s_bw)         # fusion

        # --- (2) power-line interference ---------------------------------------
        for f in (50, 100, 150):               # change to 60-Hz region if needed
            if f < nyq:
                b, a = iirnotch(f, Q=35, fs=fs)
                s1   = filtfilt(b, a, s1)

        c        = pywt.wavedec(s1, "db4", level=6)
        for lvl in (2, 3):                     # bands ≈ 31-125 Hz
            if lvl < len(c) and isinstance(c[lvl], np.ndarray):
                thr = 0.1 * np.std(c[lvl])
                c[lvl] = pywt.threshold(c[lvl], thr, "soft")
        s2       = pywt.waverec(c, "db4")[:len(s1)]

        # --- (3) high-frequency / EMG noise ------------------------------------
        d        = pywt.wavedec(s2, "db4", level=6)
        sigma    = np.median(np.abs(d[-1])) / 0.6745
        lam0     = sigma * np.sqrt(2 * np.log(len(s2)))
        dclean   = [d[0]]
        for i in range(1, len(d)):
            lam = lam0 * (0.5 + 0.5 * i / 6)   # level-adaptive
            dclean.append(pywt.threshold(d[i], lam, "soft"))
        s3       = pywt.waverec(dclean, "db4")[:len(s2)]

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
# 3.  Run denoising
# -------------------------------------------------------------------
denoised, meta = denoise_all_leads(x[:, :12], fs, lead_nms)
print("Pipeline complete — 12 leads denoised.")

# -------------------------------------------------------------------
# 4.  Quick visual check (first 5 s)
# -------------------------------------------------------------------
sec  = 5
N    = min(sec * fs, len(x))
t    = np.arange(N) / fs

plt.figure(figsize=(14, 12))
for i, l in enumerate(lead_nms):
    off = i * 3
    plt.plot(t, x[:N, i] + off,  color="gray", lw=.6)
    plt.plot(t, denoised[:N, i] + off, color="blue", lw=1)
    plt.text(-.2, off, l, va="center", fontweight="bold")
plt.title("12-lead ECG — raw (gray) vs denoised (blue)")
plt.yticks([])
plt.xlabel("Time (s)")
plt.tight_layout()
plt.show()
