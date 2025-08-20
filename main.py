import wfdb
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import matplotlib.pyplot as plt
from wfdb import processing
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
# 4. Helper: Segment the signal into PQRST complexes
# -------------------------------------------------------------------
def segment_complexes(signal, r_locs, fs):
    """
    Segments the signal into individual PQRST complexes around R-peaks.

    Args:
        signal (np.ndarray): The 1D ECG signal.
        r_locs (np.ndarray): Array of R-peak indices.
        fs (int): The sampling frequency.

    Returns:
        list: A list of numpy arrays, where each array is a PQRST complex segment.
    """
    complexes = []
    # Define window size for a complex (e.g., 0.2s before R, 0.4s after)
    pre_r_samples = int(0.2 * fs)
    post_r_samples = int(0.4 * fs)

    for r_idx in r_locs:
        start = max(0, r_idx - pre_r_samples)
        end = min(len(signal), r_idx + post_r_samples)

        # Pad with zeros if the segment is shorter than expected
        complex_segment = np.zeros(pre_r_samples + post_r_samples)
        segment_data = signal[start:end]

        # Center the segment at the R-peak
        start_pad = pre_r_samples - (r_idx - start)
        end_pad = (end - r_idx) - post_r_samples

        if start_pad < 0: start_pad = 0
        if end_pad > 0: end_pad = 0

        complex_segment[start_pad: start_pad + len(segment_data)] = segment_data

        complexes.append(complex_segment)
    return complexes


# -------------------------------------------------------------------
# 5.  Main Application Logic and GUI
# -------------------------------------------------------------------

# 5.1 Run denoising and segmentation once
denoised, meta = denoise_all_leads(x[:, :12], fs, lead_nms)
print("Pipeline complete — 12 leads denoised.")
print("Lead names available: " + ", ".join(lead_nms))

# A dictionary to store the segmented complexes for each lead
all_lead_complexes = {}
for i, l in enumerate(lead_nms):
    denoised_lead = denoised[:, i]
    xqrs = processing.XQRS(sig=denoised_lead, fs=fs)
    xqrs.detect()
    r_locs = xqrs.qrs_inds
    complexes = segment_complexes(denoised_lead, r_locs, fs)
    all_lead_complexes[l] = complexes
    print(f"Segmented {len(complexes)} complexes for lead '{l}'.")

# 5.2 Set up the GUI
root = tk.Tk()
root.title("ECG PQRST Complex Viewer")
root.geometry("1400x800")  # Set a larger default window size

# Main frame for padding
main_frame = ttk.Frame(root, padding="10")
main_frame.pack(fill=tk.BOTH, expand=True)

# UI elements frame (dropdown menu)
control_frame = ttk.Frame(main_frame)
control_frame.pack(side=tk.TOP, pady=5)

# Label and Combobox for lead selection
label = ttk.Label(control_frame, text="Select Lead:")
label.pack(side=tk.LEFT, padx=5)

lead_combo = ttk.Combobox(control_frame, values=lead_nms)
lead_combo.pack(side=tk.LEFT, padx=5)
lead_combo.set(lead_nms[0])  # Set initial selection to the first lead

# PanedWindow to divide the screen into two vertical panes
paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# Left pane for 2D plots
left_pane = ttk.Frame(paned_window, width=700)
paned_window.add(left_pane, weight=1)

# Right pane for 3D plot
right_pane = ttk.Frame(paned_window, width=700)
paned_window.add(right_pane, weight=1)

# Matplotlib figures for the plots
fig_raw_denoised = plt.figure(figsize=(10, 5))
ax1 = fig_raw_denoised.add_subplot(111)
fig_raw_denoised.set_tight_layout(True)

fig_denoised_pqrst = plt.figure(figsize=(10, 5))
ax2 = fig_denoised_pqrst.add_subplot(111)
fig_denoised_pqrst.set_tight_layout(True)

fig_3d = plt.figure(figsize=(10, 10))
ax3 = fig_3d.add_subplot(111, projection='3d')
fig_3d.set_tight_layout(True)

# Tkinter Canvases to display the matplotlib figures
canvas1 = FigureCanvasTkAgg(fig_raw_denoised, master=left_pane)
canvas_widget1 = canvas1.get_tk_widget()
canvas_widget1.pack(fill=tk.BOTH, expand=True)

canvas2 = FigureCanvasTkAgg(fig_denoised_pqrst, master=left_pane)
canvas_widget2 = canvas2.get_tk_widget()
canvas_widget2.pack(fill=tk.BOTH, expand=True)

canvas3 = FigureCanvasTkAgg(fig_3d, master=right_pane)
canvas_widget3 = canvas3.get_tk_widget()
canvas_widget3.pack(fill=tk.BOTH, expand=True)


# 5.3 Function to update all plots
def update_plot(event=None):
    """
    Updates all three plots based on the selected lead from the combo box.
    """
    selected_lead = lead_combo.get()
    lead_idx = lead_nms.index(selected_lead)

    # 10-second data
    sec = 10
    N = min(sec * fs, len(x))
    t = np.arange(N) / fs

    raw_lead = x[:N, lead_idx]
    denoised_lead_10s = denoised[:N, lead_idx]

    # PQRST complex data
    complexes = all_lead_complexes.get(selected_lead, [])
    pqrst_results = detect_pqrst(denoised[:, lead_idx], fs)

    # Clear all axes
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Plot 1: Raw vs Denoised
    ax1.plot(t, raw_lead, color="gray", lw=.6, label="Raw")
    ax1.plot(t, denoised_lead_10s, color="blue", lw=1.5, label="Denoised")
    ax1.set_title(f"Lead '{selected_lead}' — Raw vs Denoised (10s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)
    canvas1.draw()

    # Plot 2: Denoised with PQRST labels
    ax2.plot(t, denoised_lead_10s, color="blue", lw=1.5)
    ax2.plot(t[pqrst_results['P']], denoised_lead_10s[pqrst_results['P']], 'o', color='purple', markersize=4,
             label='P Peak')
    ax2.plot(t[pqrst_results['Q']], denoised_lead_10s[pqrst_results['Q']], 'o', color='green', markersize=4,
             label='Q Peak')
    ax2.plot(t[pqrst_results['R']], denoised_lead_10s[pqrst_results['R']], 'o', color='red', markersize=4,
             label='R Peak')
    ax2.plot(t[pqrst_results['S']], denoised_lead_10s[pqrst_results['S']], 'o', color='orange', markersize=4,
             label='S Peak')
    ax2.plot(t[pqrst_results['T']], denoised_lead_10s[pqrst_results['T']], 'o', color='cyan', markersize=4,
             label='T Peak')
    ax2.set_title(f"Lead '{selected_lead}' — Denoised with PQRST Complexes")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend()
    ax2.grid(True)
    canvas2.draw()

    # Plot 3: 3D view of segmented complexes
    if not complexes:
        ax3.set_title(f"No complexes found for lead '{selected_lead}'")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Complex Index")
        ax3.set_zlabel("Amplitude")
        canvas3.draw()
        return

    complex_length = len(complexes[0])
    t_complex = np.arange(complex_length) / fs

    for i, complex_waveform in enumerate(complexes[:50]):
        ax3.plot(t_complex, np.full_like(t_complex, i), complex_waveform,
                 color='blue', alpha=0.8, lw=1.5)

    ax3.set_title(f"3D View of PQRST Complexes for Lead '{selected_lead}'")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Complex Index")
    ax3.set_zlabel("Amplitude")
    ax3.view_init(elev=20, azim=-60)
    canvas3.draw()


# Bind the update function to the combobox selection event
lead_combo.bind("<<ComboboxSelected>>", update_plot)

# Initial plot display
update_plot()

# Start the main GUI loop
root.mainloop()
