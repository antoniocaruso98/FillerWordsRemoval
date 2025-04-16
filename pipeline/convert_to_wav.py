import os
import subprocess


def converti_flac_a_wav(cartella):
    # Controlla se FFmpeg è installato
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError:
        print("Errore: FFmpeg non è installato o non è presente nel PATH.")
        return

    for nome_file in os.listdir(cartella):
        #cartella_output = cartella + "_out"
        if nome_file.endswith(".flac"):
            percorso_flac = os.path.join(cartella, nome_file)
            nome_wav = os.path.splitext(nome_file)[0] + ".wav"
            percorso_wav = os.path.join(cartella, nome_wav)

            comando = [
                "ffmpeg",
                "-i",
                percorso_flac,  # Input file
                percorso_wav,  # Output file
            ]
            try:
                subprocess.run(comando, check=True)
                # print(f"Convertito: {percorso_flac} -> {percorso_wav}")
            except subprocess.CalledProcessError as e:
                print(f"Errore durante la conversione di {percorso_flac}: {e}")


# Specifica il percorso della cartella contenente i file FLAC
cartella = "./train-clean-100-flat"

converti_flac_a_wav(cartella)
