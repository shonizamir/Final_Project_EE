import torchaudio
from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import SpeakerRecognition
import torch

# --- Load separation model ---
sep_model = separator.from_hparams(source="speechbrain/sepformer-wham16k", savedir="pretrained_models/sepformer-wham16k")

# --- Load mixed audio ---
mix_wav = "mix.wav"
mix_audio, sr = torchaudio.load(mix_wav)

# --- Separate sources using SepFormer ---
est_sources = sep_model.separate_file(path=mix_wav)  # returns [2, T] waveform
est1 = est_sources[0].unsqueeze(0)  # shape: [1, T]
est2 = est_sources[1].unsqueeze(0)

# Save temporary files for comparison
torchaudio.save("est1_tmp.wav", est1, sr)
torchaudio.save("est2_tmp.wav", est2, sr)

# --- Load ground-truth speakers ---
s1_wav = "s1.wav"
s2_wav = "s2.wav"

# --- Load speaker verification model ---
spkrec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

# --- Compute similarity scores ---
score_est1_s1 = spkrec.verify_files(s1_wav, "est1_tmp.wav")
score_est1_s2 = spkrec.verify_files(s2_wav, "est1_tmp.wav")

# --- Decide best match ---
if score_est1_s1 > score_est1_s2:
    print("est1 matches s1 → labeling accordingly")
    torchaudio.save("sep_s1.wav", est1, sr)
    torchaudio.save("sep_s2.wav", est2, sr)
else:
    print("est1 matches s2 → labeling accordingly")
    torchaudio.save("sep_s1.wav", est2, sr)
    torchaudio.save("sep_s2.wav", est1, sr)

# Optional: clean up temporary files
import os
os.remove("est1_tmp.wav")
os.remove("est2_tmp.wav")