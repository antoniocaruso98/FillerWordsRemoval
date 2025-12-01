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
    return [(row['event_start_inepisode'], row['event_end_inepisode'], row['event_end_inclip'] - row['event_start_inclip']) for _, row in df.iterrows()]

def load_vad(csv_path, activation_threshold = 0.3):
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

# modificare condizione di overlap per consentire
# anche clip con minima sovrapposizione con fillers
def has_overlap(start, end, intervals):
    found = False
    too_much_overlapped = False
    for (s, e, delta_t) in intervals:
        inizio_sovrapposizione = max(start, s)
        fine_sovrapposizione = min(end, e)
        lunghezza_sovrapposizione = fine_sovrapposizione - inizio_sovrapposizione
        if lunghezza_sovrapposizione > 0.8: #0.9 * delta_t < lunghezza_sovrapposizione < 1 * delta_t:
            found = True
        #if lunghezza_sovrapposizione >= 0.4 * delta_t:
        #    too_much_overlapped = True
    return found and not too_much_overlapped


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

    for (vad_start, vad_end) in vad_intervals:
        vad_len = vad_end - vad_start

        if vad_len < TARGET_DURATION - TOLERANCE:
            continue
        
        # Se l'intervallo è troppo lungo, centra un segmento da 1s
        if vad_len >= TARGET_DURATION + TOLERANCE:
            #start = (vad_start + vad_end) / 2 - TARGET_DURATION / 2
            #end = start + TARGET_DURATION
            start = vad_start
            end = vad_start + TARGET_DURATION

            while end <= vad_end:
                actual_duration = end - start
                if not (TARGET_DURATION - TOLERANCE <= actual_duration <= TARGET_DURATION + TOLERANCE):
                    break
                if start < 0 or end > duration:
                    break
                if has_overlap(start, end, filler_intervals):
                    start += TARGET_DURATION
                    end += TARGET_DURATION
                    continue

                # Salva il segmento
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                crop = y[start_sample:end_sample]

                output_filename = f"{file_id}_{int(start*1000)}ms.wav"
                output_path = os.path.join(OUTPUT_DIR, output_filename)
                sf.write(output_path, crop, sr)
                print(f'saved {output_filename}')

                start += TARGET_DURATION
                end += TARGET_DURATION
            continue
        else:
            # Usa tutto l'intervallo (dentro la tolleranza)
            start = vad_start
            end = vad_end
        

        #start = vad_start
        #end = vad_start + TARGET_DURATION - TOLERANCE

        #while(end <= vad_end):
        actual_duration = end - start
        if not (TARGET_DURATION - TOLERANCE <= actual_duration <= TARGET_DURATION + TOLERANCE):
            continue
        if start < 0 or end > duration:
            continue
        if not has_overlap(start, end, filler_intervals):
            continue

        # Salva il segmento
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        crop = y[start_sample:end_sample]

        output_filename = f"{file_id}_{int(start*1000)}ms.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        sf.write(output_path, crop, sr)
        print(f'saved {output_filename}')

            # Prossimo segmento
            #start += TARGET_DURATION - TOLERANCE
            #end += TARGET_DURATION - TOLERANCE