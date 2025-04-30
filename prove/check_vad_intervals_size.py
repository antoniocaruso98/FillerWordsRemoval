import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm

dataset = 'validation'

# === Percorsi dataset ===
AUDIO_DIR = r"D:\Podcast_filler_dataset\PodcastFillers\PodcastFillers\audio\episode_wav" + f"\{dataset}"
ANNOTATIONS_DIR = r"D:\Podcast_filler_dataset\PodcastFillers\PodcastFillers\metadata\episode_annotations" + f"\{dataset}"
VAD_DIR = r"D:\Podcast_filler_dataset\PodcastFillers\PodcastFillers\metadata\episode_vad" + f"\{dataset}"
#OUTPUT_DIR = r"D:\Podcast_filler_dataset\PodcastFillers\PodcastFillers\clean_segments"
OUTPUT_DIR = r"clean_segments_v2" + f"\\{dataset}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Parametri del crop ===
TARGET_DURATION = 1.0       # secondi
TOLERANCE = 0.1             # ±100ms

def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    return [(row['event_start_inepisode'], row['event_end_inepisode']) for _, row in df.iterrows()]

def load_vad(csv_path, activation_threshold = 0.5):
    df = pd.read_csv(csv_path, header=None, names=['time', 'activation'])
    vad_intervals = []
    start = None
    for _, row in df.iterrows():
        if row['activation'] >= activation_threshold:
            if start is None:
                start = row['time']
        else:
            if start is not None:
                vad_intervals.append((start, row['time']))
                start = None
    if start is not None:
        vad_intervals.append((start, df['time'].iloc[-1]))
    return vad_intervals

def has_overlap(start, end, intervals):
    for (s, e) in intervals:
        if start < e and end > s:
            return True
    return False

# === Elaborazione ===
for file in tqdm(os.listdir(AUDIO_DIR)):
    
    print(f'processing file {file}')
    
    if not file.endswith(".wav"):
        continue

    file_id = os.path.splitext(file)[0]
    audio_path = os.path.join(AUDIO_DIR, file)
    annotation_path = os.path.join(ANNOTATIONS_DIR, file_id + ".csv")
    vad_path = os.path.join(VAD_DIR, file_id + ".csv")

    if not os.path.exists(annotation_path) or not os.path.exists(vad_path):
        continue

    # Carica audio e info
    y, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=y, sr=sr)
    filler_intervals = load_annotations(annotation_path)
    vad_intervals = load_vad(vad_path)

    max = 0
    nr_ge_2000 = 0

    for (vad_start, vad_end) in vad_intervals:
        vad_len = (vad_end - vad_start) * 1000
        if vad_len >= 2000:
            nr_ge_2000 += 1
        if vad_len > max:
            max = vad_len
    
    print(f'max interval length: {max} ms')
    print(f'number of long intervals: {nr_ge_2000}')
