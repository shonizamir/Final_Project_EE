import os
import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi
from measurements import get_measurements
from resemblyzer import preprocess_wav, VoiceEncoder
from scipy.spatial.distance import cosine
from scipy.signal import correlate


def waveform_similarity(a, b):
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)
    corr = correlate(a, b, mode='valid')
    return np.max(corr)


base_path = "/storage/shoni/UniAudio/UniAudio/egs/SPEX/exp/results/inference_spex_inference/spex_test/0"
counter = 0
num_errors_vals = 0
num_errors_sim = 0
different_recs = 0
# Initialize metric containers
si_sdr_list, pesq_list, stoi_list = [], [], []

# Load Resemblyzer encoder
encoder = VoiceEncoder()

# Scan for all sampling_sample0 files
all_files = os.listdir(base_path)
sample_files = [f for f in all_files if f.endswith("sampling_sample0.wav")]

for sample_file in sample_files:
    counter += 1
    try:
        base_id = sample_file.replace("_sampling_sample0.wav", "")
        input2_file = f"{base_id}_input2.wav"

        sample_path = os.path.join(base_path, sample_file)
        input2_path = os.path.join(base_path, input2_file)

        if not os.path.exists(input2_path):
            print(f"Warning: Input2 file not found for {sample_file}")
            continue

        # Load files
        x, sr_x = sf.read(sample_path)
        y, sr_y = sf.read(input2_path)

        if sr_x != sr_y:
            raise ValueError(f"Sampling rates do not match: {sr_x} != {sr_y}")

        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]

        # Speaker similarity check
        sample_audio = preprocess_wav(sample_path)
        input2_audio = preprocess_wav(input2_path)

        sample_embed = encoder.embed_utterance(sample_audio)
        input2_embed = encoder.embed_utterance(input2_audio)
        similarity = 1 - cosine(sample_embed, input2_embed)
        if similarity < 0.8:  # Threshold can be tuned
            num_errors_sim += 1

        # Compute metrics
        si_sdr_val, pesq_val, stoi_val = get_measurements(input2_path, sample_path)
        
        if si_sdr_val < 3:  # Optional: waveform similarity check
            num_errors_vals +=1 

        waveform_sim = waveform_similarity(x, y)
        if waveform_sim < 0.8:
            different_recs += 1
        # Accumulate metrics
        si_sdr_list.append(si_sdr_val)
        pesq_list.append(pesq_val)
        stoi_list.append(stoi_val)

        print(f"{base_id}: Sim={similarity:.2f}, SI-SDR={si_sdr_val:.2f}, PESQ={pesq_val:.2f}, ESTOI={stoi_val:.2f}")

    except Exception as e:
        print(f"❌ Error in {sample_file}: {e}")
        raise

# Results summary
def mean_std_str(arr):
    arr = np.array(arr)
    arr = arr[~np.isnan(arr)]
    return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"

print("\n===== AVERAGE METRICS =====")
print(f"SI-SDR: {mean_std_str(si_sdr_list)} dB")
print(f"PESQ: {mean_std_str(pesq_list)}")
print(f"ESTOI: {mean_std_str(stoi_list)}")
print(f"Total amount of files validated: {counter}")
print(f"Total diff: {different_recs}, err_sim: {num_errors_sim}, num_errors_vals: {num_errors_vals}")