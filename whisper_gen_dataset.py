import whisper
import os
from pydub import AudioSegment

# Percorso al file
audio_folder = 'dev-clean-flatten'

# Durata massima del campione (1s)
MAX_DURATION_MS = 1000

# Estrai solo le frasi che stanno in 1 secondo e che
# sono più lunghi di MAX_DURATION_MS meno una certa soglia eps in ms
eps = 200

# Carica il modello Whisper
model = whisper.load_model("small")

for audio_file in os.listdir(audio_folder):
    # Trascrivi con segmenti frase
    result = model.transcribe(os.path.join(audio_folder,audio_file), language="en", word_timestamps=True)

    audio = AudioSegment.from_wav(os.path.join(audio_folder,audio_file))

    for i, segment in enumerate(result['segments']):
        start_ms = int(segment['start'] * 1000)
        end_ms = int(segment['end'] * 1000)
        duration = end_ms - start_ms

        if duration <= MAX_DURATION_MS and duration >= MAX_DURATION_MS - eps:
            sample = audio[start_ms:end_ms]
            filename = f"{audio_file}_{i}.wav"
            sample.export(filename, format="wav")
            print(f"Salvato: {filename} - '{segment['text']}'")