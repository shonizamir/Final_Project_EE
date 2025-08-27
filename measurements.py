import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from pesq import pesq
from pystoi import stoi
import pyloudnorm as pyln
from scipy.signal import correlate
from scipy.interpolate import interp1d
import librosa
from speechmos import dnsmos



def get_dnsmos(file_path: str) -> dict:
    """
    Compute DNSMOS for a single audio file.

    Args:
        file_path: Path to a mono audio file.

    Returns:
        A dict with keys:
          - 'ovrl_mos': overall MOS
          - 'sig_mos': signal quality MOS
          - 'bak_mos': background quality MOS
          - 'p808_mos': P.808-style MOS (if available)
    """
    # load at 16 kHz mono
    audio, sr = librosa.load(file_path, sr=16000, mono=True)
    # run DNSMOS
    scores = dnsmos.run(audio, sr)
    return scores

def loudness_normalize(audio, sr):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)
    return pyln.normalize.loudness(audio, loudness, -23.0)

def dtw_align_stft(clean, noisy, sr, win_length=512, hop_length=256):
    print("[Debug] Computing STFT magnitude for DTW alignment...")
    clean_stft = np.abs(librosa.stft(clean, n_fft=win_length, hop_length=hop_length))
    noisy_stft = np.abs(librosa.stft(noisy, n_fft=win_length, hop_length=hop_length))

    D, wp = librosa.sequence.dtw(X=clean_stft, Y=noisy_stft, metric='euclidean')
    wp = np.array(wp)[::-1]

    ref_frames = wp[:, 0]
    test_frames = wp[:, 1]

    ref_samples = ref_frames * hop_length
    test_samples = test_frames * hop_length

    _, unique_indices = np.unique(ref_samples, return_index=True)
    ref_samples_unique = ref_samples[unique_indices]
    test_samples_unique = test_samples[unique_indices]

    if len(ref_samples_unique) < 2:
        print("[Error] Not enough unique samples after DTW to perform interpolation.")
        return np.zeros_like(clean)

    interp_func = interp1d(ref_samples_unique, test_samples_unique,
                           kind='linear', fill_value="extrapolate", bounds_error=False)

    aligned = np.zeros_like(clean)
    for i in range(len(clean)):
        noisy_idx = interp_func(i)
        if np.isnan(noisy_idx):
            continue
        noisy_idx = np.clip(noisy_idx, 0, len(noisy) - 1)
        aligned[i] = np.interp(noisy_idx, np.arange(len(noisy)), noisy)

    return aligned

def si_sdr(s, s_hat):
    alpha = np.dot(s_hat, s) / np.linalg.norm(s)**2
    return 10 * np.log10(np.linalg.norm(alpha * s)**2 / np.linalg.norm(alpha * s - s_hat)**2)

def print_stats(x, label):
    print(f"[Stats] {label}: mean={np.mean(x):.4f}, std={np.std(x):.4f}, max={np.max(np.abs(x)):.4f}")

def plot_waveforms(clean, noisy, title_suffix=""):
    plt.figure(figsize=(12, 4))
    plt.plot(clean, label='Clean', alpha=0.7)
    plt.plot(noisy, label='Noisy', alpha=0.7)
    plt.title(f'Waveforms {title_suffix}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spectrograms(clean, noisy, sr, title_suffix=""):
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.specgram(clean, Fs=sr)
    plt.title(f'Clean Spectrogram {title_suffix}')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.specgram(noisy, Fs=sr)
    plt.title(f'Noisy Spectrogram {title_suffix}')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def get_measurements(clean_path, noisy_path, chunk_dur=1.0, sr=16000, min_sdr=-20, use_dtw=True, plot=True):
    clean, _ = librosa.load(clean_path, sr=sr)
    noisy, _ = librosa.load(noisy_path, sr=sr)

    min_len = min(len(clean_path), len(noisy_path))
    clean_aligned = clean_path[:min_len]
    noisy_aligned = noisy_path[:min_len]


    si_sdr_val = si_sdr(clean_path, noisy_path)
    p = pesq(clean_path, noisy_path)
    s = stoi(clean_path, noisy_path)
    d = get_dnsmos(clean_path, noisy_path)
    print(f"\nChunk-wise averaged metrics over {len(sdr_list)} chunk(s) of {chunk_dur} sec:")
    print(f"SI-SDR: {si_sdr_val:.2f} dB")
    print(f"PESQ: {p:.2f}")
    print(f"ESTOI: {s:.2f}")
    print(f"DNSMOS: {d:.2f}")

    return si_sdr_val, p, s, d

