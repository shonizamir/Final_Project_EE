#!/usr/bin/env python3
import os, re
import glob
import sys
import librosa
import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore  # DNSMOS via TorchMetrics
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ——————————— Helper functions ——————————— #


def plot_and_save_specs_and_waveforms(name, clean_path, noisy_path, sep_path, output_dir="audio_plots_simplex-final"):
    """
    - clean_path: path to the clean reference WAV
    - noisy_path: path to the noisy/enhanced WAV
    - output_dir: directory where images will be saved
    """
    # 1. Print file locations
    print(f"Clean file: {clean_path}")
    print(f"Noisy file: {noisy_path}")

    # 2. Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)
    t_sr = 8000
    # 3. Load audio
    y_clean, sr_clean = librosa.load(clean_path, sr=None)
    y_noisy, sr_noisy = librosa.load(noisy_path, sr=None)
    y_sep, sr_sep = librosa.load(sep_path, sr=None)

    
    if sr_clean != t_sr:
        print(f"[WARN] Resampling noisy from {sr_noisy}→{t_sr}")
        y_clean = librosa.resample(y_clean, orig_sr=sr_clean, target_sr=t_sr)
        sr_clean = t_sr
    # 4. Resample noisy if necessary
    if sr_noisy != t_sr:
        print(f"[WARN] Resampling noisy from {sr_noisy}→{t_sr}")
        y_noisy = librosa.resample(y_noisy, orig_sr=sr_noisy, target_sr=t_sr)
        sr_noisy = t_sr
        
    if sr_sep != sr_clean:
        print(f"[WARN] Resampling noisy from {sr_sep}→{t_sr}")
        y_sep = librosa.resample(y_sep, orig_sr=sr_sep, target_sr=sr_clean)
        sr_sep = t_sr
    # 5. Compute spectrograms (magnitude → dB)
    S_clean    = np.abs(librosa.stft(y_clean))
    S_clean_db = librosa.amplitude_to_db(S_clean, ref=np.max)
    S_noisy    = np.abs(librosa.stft(y_noisy))
    S_noisy_db = librosa.amplitude_to_db(S_noisy, ref=np.max)
    S_sep    = np.abs(librosa.stft(y_sep))
    S_sep_db = librosa.amplitude_to_db(S_sep, ref=np.max)

    # 6. Save clean spectrogram
    clean_spec_file = os.path.join(output_dir, name + "_clean_spectrogram.png")
    plt.figure()
    librosa.display.specshow(S_clean_db, sr=t_sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Original Clean Spectrogram (GT)")
    plt.savefig(clean_spec_file)
    plt.close()

    # 7. Save noisy spectrogram
    noisy_spec_file = os.path.join(output_dir, name + "_noisy_spectrogram.png")
    plt.figure()
    librosa.display.specshow(S_noisy_db, sr=t_sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Output Spectrogram")
    plt.savefig(noisy_spec_file)
    plt.close()
    
    sep_spec_file = os.path.join(output_dir, name + "_simplex_spectrogram.png")
    plt.figure()
    librosa.display.specshow(S_sep_db, sr=t_sr, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Simplex Spectrogram")
    plt.savefig(sep_spec_file)
    plt.close()

    # 8. Plot waveforms together
    #    time axes
    t_clean = np.linspace(0, len(y_clean) / sr_clean, len(y_clean))
    t_noisy = np.linspace(0, len(y_noisy) / sr_noisy, len(y_noisy))

    


    t_sep = np.linspace(0, len(y_sep) / sr_sep, len(y_sep))
    waveform_file = os.path.join(output_dir,name + "simplex_waveforms.png")
    plt.figure()
    plt.plot(t_clean, y_clean, label="GT", color="blue", alpha=0.6)
    plt.plot(t_sep, y_sep, label="Simplex", color="red", alpha=0.6)
    plt.legend()
    plt.title("Waveforms (Gt vs Simplex)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(waveform_file)
    plt.close()
    

    waveform_file = os.path.join(output_dir,name + "_waveforms.png")
    plt.figure()
    plt.plot(t_clean, y_clean, label="GT", color="blue", alpha=0.6)
    plt.plot(t_noisy, y_noisy, label="Output", color="red", alpha=0.6)
    plt.legend()
    plt.title("Waveforms (GT vs Output)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(waveform_file)
    plt.close()
    




def load_wav(path):
    """Load a WAV file and return (signal, sample_rate)."""
    sig, sr = sf.read(path)
    if sig.ndim > 1:
        sig = sig[:, 0]
    return sig.astype(np.float64), sr


def compute_si_sdr(ref, est, eps=1e-8):
    ref_zm = ref - np.mean(ref)
    est_zm = est - np.mean(est)
    alpha  = np.sum(est_zm * ref_zm) / (np.sum(ref_zm**2) + eps)
    target = alpha * ref_zm
    noise  = est_zm - target
    return 10 * np.log10((np.sum(target**2) + eps) / (np.sum(noise**2) + eps))


def compute_sd_sdr(ref, est, eps=1e-8):
    ref_zm = ref - np.mean(ref)
    est_zm = est - np.mean(est)
    noise  = est_zm - ref_zm
    return 10 * np.log10((np.sum(ref_zm**2) + eps) / (np.sum(noise**2) + eps))


MIN_DURATION_S = 0.2    # skip clips shorter than this
MIN_ENERGY    = 1e-6    # skip clips whose RMS energy is below this

def compute_pesq(ref: np.ndarray, est: np.ndarray, fs: int) -> float:
    """
    Compute PESQ in narrowband (8 kHz) or wideband (16 kHz) mode,
    but if anything goes wrong (silent clip, too short, or PESQ bug),
    return np.nan instead of raising.
    """
    # 1) quick length/energy filter
    if len(ref) == 0:
        return 0
    duration = len(ref) / fs
    if duration < MIN_DURATION_S:
        return 0
    rms_ref = np.sqrt(np.mean(ref**2))
    if rms_ref < MIN_ENERGY:
        return 0

    # 2) choose mode
    mode = 'nb' if fs == 8000 else 'wb'

    # 3) call PESQ, catch absolutely everything
    try:
        # if your PESQ version supports on_error, this silences internal raises
        return pesq(fs, ref, est, mode, on_error='warn')
    except TypeError:
        # older PESQ wheels will blow up on on_error kwarg
        try:
            return pesq(fs, ref, est, mode)
        except Exception as e:
            print(f"[WARN] PESQ(full) failed: {e}")
            return 0
    except Exception as e:
        print(f"[WARN] PESQ failed: {e}")
        return 0


def compute_estoi(ref, est, sr):
    return stoi(ref, est, sr, extended=True)



def main(base_dir):
    # ——————————— Main routine ——————————— #


    # Load scp mappings
    wav_dict = {}
    noise_dict = {}

    with open("/storage/danielk/UniAudio/UniAudio/egs/SE/data/simplex_final_5s/test/1splits/wav.1.scp", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                key, path = parts
                wav_dict[key] = path

    with open("/storage/danielk/UniAudio/UniAudio/egs/SE/data/simplex_final_5s/test/1splits/noise.1.scp", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                key, path = parts
                noise_dict[key] = path
    # find all sampling files
    ref_dir = "/storage/shoni/UniAudio/UniAudio/egs/SE/audio-updated"
    pattern = os.path.join(base_dir, "*_sampling_sample0.wav")
    sampling_list = sorted(glob.glob(pattern))
    if not sampling_list:
        print("No files matching *_sampling_sample0.wav found in", base_dir)
        return

    # accumulators for metrics
    metrics = {
        'SI-SDR' : [],
        'SD-SDR' : [],
        'PESQ'   : [],
        'ESTOI'  : [],
        'DNSMOS' : [],
    }
    
    metrics_sepFormer = {
        'SI-SDR' : [],
        'SD-SDR' : [],
        'PESQ'   : [],
        'ESTOI'  : [],
        'DNSMOS' : [],
    }

    dns_metric = None  # will initialize after we know sample rate
    dns_metric_sep = None 
    
    for samp_path in sampling_list:
        #t_sr = 16000
        t_sr = 8000
        print(f"samp_path: {samp_path}\n")
        fn = os.path.basename(samp_path)
        # Extract recnum key
        match = re.search(r'(recnum_\d+_\d+_\d+)', fn)
        if not match:
            print(f"[WARN] No valid recnum key found in {fn} – skipping")
            continue
        rec_key = match.group(1)

        if rec_key not in wav_dict or rec_key not in noise_dict:
            print(f"[WARN] recnum key {rec_key} not found in .scp files – skipping")
            continue

        inp1_path = wav_dict[rec_key]   # clean (GT)
        inp0_path = noise_dict[rec_key] # noisy (input to model)
        sepformer_path = inp0_path
        if not os.path.isfile(inp1_path):
            print(f"[WARN] missing {os.path.basename(inp1_path)} – skipping")
            continue
        
        plot_and_save_specs_and_waveforms(fn, inp1_path, samp_path, sepformer_path)

        # load signals
        ref, sr1 = load_wav(inp1_path)
        sep_est, sr3 = load_wav(sepformer_path)
        est, sr2 = load_wav(samp_path)

        sep_est = librosa.resample(sep_est, orig_sr=sr3, target_sr=t_sr)
        est = librosa.resample(est, orig_sr=sr2, target_sr=t_sr)
        ref = librosa.resample(ref, orig_sr=sr1, target_sr=t_sr)
        # initialize DNSMOS metric on first iteration
        if dns_metric is None:
            dns_metric = DeepNoiseSuppressionMeanOpinionScore(fs=sr1, personalized=False)

        if dns_metric_sep is None:
            dns_metric_sep = DeepNoiseSuppressionMeanOpinionScore(fs=sr1, personalized=False)
        # trim to same length
        min_len = min(len(ref), len(est), len(sep_est))
        ref = ref[:min_len]
        est = est[:min_len]
        sep_est = sep_est[:min_len]

        # compute metrics
        si_val  = compute_si_sdr(ref, est)
        sd_val  = compute_sd_sdr(ref, est)
        psq_val = compute_pesq(ref, est, t_sr)
        et_val  = compute_estoi(ref, est, t_sr)
        
        si_val_sep  = compute_si_sdr(ref, sep_est)
        sd_val_sep  = compute_sd_sdr(ref, sep_est)
        psq_val_sep = compute_pesq(ref, sep_est, t_sr)
        et_val_sep  = compute_estoi(ref, sep_est, t_sr)

        # compute DNSMOS (overall MOS)
        est_tensor = torch.from_numpy(est).unsqueeze(0)  # shape [1, time]
        with torch.no_grad():
            dns_scores = dns_metric(est_tensor)  # could be 1D or 2D
        # handle both shapes:
        if dns_scores.ndim == 2:
            ovr_val = dns_scores[0, 3].item()
        else:
            ovr_val = dns_scores[3].item()
            
        
        est_sep_tensor = torch.from_numpy(sep_est).unsqueeze(0)  # shape [1, time]
        with torch.no_grad():
            dns_scores_sep = dns_metric(est_sep_tensor)  # could be 1D or 2D
        # handle both shapes:
        if dns_scores_sep.ndim == 2:
            ovr_val_sep = dns_scores_sep[0, 3].item()
        else:
            ovr_val_sep = dns_scores_sep[3].item()

        # store
        metrics['SI-SDR'].append(si_val)
        metrics['SD-SDR'].append(sd_val)
        metrics['PESQ'].append(psq_val)
        metrics['ESTOI'].append(et_val)
        metrics['DNSMOS'].append(ovr_val)
        
        metrics_sepFormer['SI-SDR'].append(si_val_sep)
        metrics_sepFormer['SD-SDR'].append(sd_val_sep)
        metrics_sepFormer['PESQ'].append(psq_val_sep)
        metrics_sepFormer['ESTOI'].append(et_val_sep)
        metrics_sepFormer['DNSMOS'].append(ovr_val_sep)

        # print per-file
        print(f"{fn} gt vs pred: SI-SDR={si_val:6.2f} dB  SD-SDR={sd_val:6.2f} dB  "
              f"PESQ={psq_val:.2f}  ESTOI={et_val:.2f}  DNSMOS={ovr_val:.2f}")
        
        print(f"{fn}  gt vs sepFormer: SI-SDR={si_val:6.2f} dB  SD-SDR={sd_val:6.2f} dB  "
              f"PESQ={psq_val:.2f}  ESTOI={et_val:.2f}  DNSMOS={ovr_val:.2f}")

    # summary
    print("\n=== AVERAGE OVER ALL FILES - GT VS PRED===")
    for key, vals in metrics.items():
        if vals:
            mean = np.mean(vals)
            std  = np.std(vals, ddof=1)
            print(f"{key:6s}: {mean:6.2f} ± {std:5.2f}")
        else:
            print(f"{key:6s}: no data")
            
            
    print("\n=== AVERAGE OVER ALL FILES - GT VS SIMPLEX===")
    for key, vals in metrics_sepFormer.items():
        if vals:
            mean = np.mean(vals)
            std  = np.std(vals, ddof=1)
            print(f"{key:6s}: {mean:6.2f} ± {std:5.2f}")
        else:
            print(f"{key:6s}: no data")

if __name__ == "__main__":
    default_dir = ("/storage/shoni/UniAudio/UniAudio/egs/SE/exp/simplex_5s_final/inference_se_inference/se_test/0")
    base = sys.argv[1] if len(sys.argv) > 1 else default_dir
    main(base)

