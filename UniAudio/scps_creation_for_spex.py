import os
import argparse
import random
import shutil
from collections import defaultdict

def get_speaker_id_from_filename(fname):
    parts = fname.split('-')
    if fname.startswith('s1-'):
        return parts[1]  # s1-84-...
    elif fname.startswith('s2-'):
        parts[3].split('_')[1]
        print("file name is ", fname, flush=True)
        print("speaker is is ", parts[3].split('_')[1], flush=True)
        return parts[3].split('_')[1]  # s2-84-121123... => 5895
    return None

def remove_suffix_files(folder):
    for fname in os.listdir(folder):
        if fname.endswith('sepFormer.wav') or fname.endswith('prompt.wav'):
            os.remove(os.path.join(folder, fname))

def collect_clean_files(folder):
    """
    Returns:
    - entries: List of dicts with keys: key, speaker_id, clean_path, mixed_path
    - speaker_to_files: Dict from speaker ID to list of clean paths (excluding the file itself)
    """
    entries = []
    speaker_to_files = defaultdict(list)

    files = os.listdir(folder)
    clean_files = [f for f in files if f.startswith('s1-') or f.startswith('s2-')]

    for fname in clean_files:
        full_clean_path = os.path.join(folder, fname)
        key = fname[:-4]  # remove .wav
        speaker_id = get_speaker_id_from_filename(fname)
        mixed_name = fname.split('-', 1)[1]  # remove s1-/s2-
        mixed_path = os.path.join(folder, mixed_name)

        if not os.path.exists(mixed_path):
            print(f"Skipping {fname} - no matching mixed file found.")
            continue

        entries.append({
            'key': key,
            'speaker_id': speaker_id,
            'clean_path': full_clean_path,
            'mixed_path': mixed_path
        })
        speaker_to_files[speaker_id].append(full_clean_path)

    return entries, speaker_to_files

def choose_prompt(entry, speaker_to_files):
    """
    Chooses a different utterance for the same speaker.
    Returns: path to prompt file or None
    """
    options = [p for p in speaker_to_files[entry['speaker_id']] if p != entry['clean_path']]
    if not options:
        return None
    return random.choice(options)

def write_scp_file(path, entries, key_field, value_field):
    with open(path, 'w') as f:
        for entry in entries:
            f.write(f"{entry[key_field]} {entry[value_field]}\n")

def write_utt2spk(path, entries, key_field, prompt_field):
    with open(path, 'w') as f:
        for entry in entries:
            f.write(f"{entry[key_field]} {entry[prompt_field]}\n")

def split_entries(entries, val_ratio=0.1):
    random.shuffle(entries)
    split_point = int(len(entries) * (1 - val_ratio))
    return entries[:split_point], entries[split_point:]

def process_all(input_folder, output_folder, val_ratio=0.1):
    os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)

    remove_suffix_files(input_folder)
    entries, speaker_to_files = collect_clean_files(input_folder)

    filtered_entries = []
    for entry in entries:
        prompt_path = choose_prompt(entry, speaker_to_files)
        if prompt_path is None:
            print(f"No prompt found for {entry['key']}, skipping...")
            continue
        entry['prompt_path'] = prompt_path
        filtered_entries.append(entry)

    train_entries, val_entries = split_entries(filtered_entries, val_ratio=val_ratio)

    def dump_set(set_entries, split_name):
        split_dir = os.path.join(output_folder, split_name)
        write_scp_file(os.path.join(split_dir, 'clean.scp'), set_entries, 'key', 'clean_path')
        write_scp_file(os.path.join(split_dir, 'noise.scp'), set_entries, 'key', 'mixed_path')
        write_utt2spk(os.path.join(split_dir, 'utt2spk'), set_entries, 'key', 'prompt_path')

    dump_set(train_entries, 'train')
    dump_set(val_entries, 'val')

    print(f"✅ Done. Train entries: {len(train_entries)} | Val entries: {len(val_entries)}")

def main():
    parser = argparse.ArgumentParser(description="Generate scp files for speech extraction training.")
    input_folder = '/storage/shoni/UniAudio/UniAudio/egs/SPEX/audio_prompts'
    output_folder = '/storage/shoni/UniAudio/UniAudio/egs/SPEX/data'
    val_ratio = 0.1

    process_all(input_folder, output_folder, val_ratio)

if __name__ == "__main__":
    main()
