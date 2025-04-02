import sys
import os
import pydub #da scaricare con pip + pip install imageio[ffmpeg]
from pydub import AudioSegment


def split_audio(input_file=".\ytmp3free.cc_zia-tina-compilation-bella-cadduosa-e-speciali-e-youtubemp3free.org.mp3",
                 clip_length_s=10):
    # docstring di info che chiamo con help(split_audio)
    '''
    Divide un file audio di formato e lunghezza arbitrari
    in una serie di clip di lunghezza specifiata in ms.
    Es: clip_length = 10s = 10 * 1000 ms.
    Se ultima clip è minore della lunghezza desiderata, padding
    con silenzio.

    Args:
        input_file (str): Il percorso del file audio di input. Default: "input.mp3".
        clip_length_s (int): La lunghezza desiderata di ogni clip in secondi. Default: 10.
    Returns:
        None: Le clip vengono salvate come file separati nel formato mp3.

    Da modificare l'organizzazione in sottocartelle delle clip di ogni audio.
    '''

    if not os.path.exists(input_file):
        print(f"Errore: Il file '{input_file}' non esiste.")
        return
    # Carica il file audio (sostituisci 'input.mp3' con il tuo file)
    audio = AudioSegment.from_file(input_file)

    # Imposta la durata delle clip in millisecondi (10 secondi)
    clip_length = clip_length_s * 1000  
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



# esecuzione da terminale: python generate_clips_prova.py {args}
if __name__ == "__main__":
    split_audio()