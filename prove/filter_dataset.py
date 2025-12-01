import os
import re
import shutil
import whisper

def is_clip_representative(file_path, tolerance=0.05, center_tolerance=0.1):
    """
    Verifica se una clip da 1 secondo è rappresentativa:
      - La prima parola deve iniziare entro 'tolerance' secondi dall'inizio (idealmente quasi 0).
      - L'ultima parola deve terminare entro 'tolerance' secondi dal termine del clip (idealmente quasi 1).
      - Il centro del parlato (media tra inizio della prima e fine dell'ultima parola) deve trovarsi vicino a 0.5 sec (entro 'center_tolerance').
    Restituisce True se tutte le condizioni sono verificate, altrimenti False.
    """
    try:
        result = model.transcribe(file_path, language="en", word_timestamps=True, verbose=False)
    except Exception:
        return False

    if 'segments' not in result or len(result['segments']) == 0:
        return False

    # Consideriamo il primo segmento, che dovrebbe coprire l'intera clip da 1s
    segment = result['segments'][0]
    if 'words' not in segment or len(segment['words']) == 0:
        return False

    words = segment['words']
    first_word_start = words[0]['start']  # in secondi
    last_word_end = words[-1]['end']        # in secondi

    # Controlla che la prima parola inizi quasi all'inizio (entro il margine di tolleranza)
    if first_word_start > tolerance:
        return False

    # Controlla che l'ultima parola termini quasi a 1 secondi (entro il margine di tolleranza)
    if (1.0 - last_word_end) > tolerance:
        return False

    # Verifica che il centro del parlato sia vicino a 0.5 sec
    spoken_center = (first_word_start + last_word_end) / 2.0
    if abs(spoken_center - 0.5) > center_tolerance:
        return False

    return True

# Percorsi delle cartelle (modifica se necessario)
input_folder = os.path.join("..",os.path.join("Dataset_completo","train"))              # Cartella contenente i file .wav da 1 s
output_folder = os.path.join("..","train_clips") # Cartella in cui verranno copiati i file validi

# Crea la cartella di output se non esiste
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Carica il modello Whisper (qui si usa la versione "small")
model = whisper.load_model("small")

# Itera su ogni file .wav nella cartella di input
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(".wav"):
        file_path = os.path.join(input_folder, file_name)
        base_name, _ = os.path.splitext(file_name)
        # Se il nome del file è formato da esattamente 5 cifre, copialo direttamente
        if re.match(r"^\d{5}$", base_name):
            shutil.copy2(file_path, os.path.join(output_folder, file_name))
            continue

        # Altrimenti, applica i controlli per la rappresentatività
        try:
            if is_clip_representative(file_path):
                shutil.copy2(file_path, os.path.join(output_folder, file_name))
        except Exception:
            pass  # In caso di errori, il file viene saltato senza stampare nulla
