import sys
import librosa
import numpy as np
from pydub import AudioSegment

# Raggruppa gli indici continui di silenzio in intervalli
def find_silent_intervals(indices):
    if len(indices) == 0:
        return []
    
    intervals = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i-1] + 1:
            intervals.append((start, indices[i-1]))
            start = indices[i]
    intervals.append((start, indices[-1]))
    
    return intervals

def main(audio_path):
    # Carica il file audio
    audio, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio caricato correttamente da {audio_path}")
    print(f"Durata: {len(audio)/sr:.2f} secondi, Frequenza di campionamento: {sr} Hz")
    
    power= librosa.feature.rms(y=audio).flatten()
    times = librosa.times_like(power, sr=sr)

    threshold = min(power) + 0.1 * (max(power) - min(power))
    # Trova gli indici in cui il valore di RMS è inferiore alla soglia
    silent_indices = np.where(power < threshold)

    silent_intervals = find_silent_intervals(silent_indices)

    # Converti gli intervalli in tempo
    time_intervals = [(times[start], times[end]) for start, end in silent_intervals] # [s]

    print("Intervalli di silenzio:", time_intervals)

    silence_duration_threshold = 0.150  # Durata minima del silenzio naturale in s 
    long_silent_intervals = []
    for start, end in time_intervals:
        duration = end - start
        if duration > silence_duration_threshold:
            long_silent_intervals.append((start, end)) # effective silence

    audio_segment = AudioSegment.from_file(audio_path)
    audio_clean= []
    for start, end in long_silent_intervals:
        start_ms = int(start * 1000)  # Converti in millisecondi
        end_ms = int(end * 1000)      # Converti in millisecondi
        audio_clean.append(audio_segment[:start_ms]+ AudioSegment.silent(duration=silence_duration_threshold)+ audio_segment[end_ms:]) 
    audio_clean = np.concatenate(audio_clean)
    audio_clean.export("audio_clean.wav", format="wav") 
            


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Utilizzo: python inference.py <path_del_file_audio>")
        sys.exit(1)
    audio_path = sys.argv[1]
    main(audio_path)

