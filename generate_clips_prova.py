'''
Divide un file audio di formato e lunghezza arbitrari
in una serie di clip di lunghezza specifiata in ms.
Es: clip_length = 10s = 10 * 1000 ms.
Se ultima clip è minore della lunghezza desiderata, padding
con silenzio.

Da modificare l'organizzazione in sottocartelle delle clip di ogni audio.
'''

from pydub import AudioSegment

# Carica il file audio (sostituisci 'input.mp3' con il tuo file)
audio = AudioSegment.from_file("input.mp3")

# Imposta la durata delle clip in millisecondi (10 secondi)
clip_length = 10 * 1000  
total_length = len(audio)

# Creazione delle clip
for i in range(0, total_length, clip_length):
    clip = audio[i:i + clip_length]

    # Se l'ultima clip è più corta di 10 secondi, aggiungi silenzio
    if len(clip) < clip_length:
        silence_duration = clip_length - len(clip)
        clip += AudioSegment.silent(duration=silence_duration)

    # Salva la clip
    clip.export(f"clip_{i // clip_length + 1}.mp3", format="mp3")

print("Divisione completata!")
