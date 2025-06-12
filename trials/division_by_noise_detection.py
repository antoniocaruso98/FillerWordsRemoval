import os
import re
import shutil
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def extract_representative_clip(file_path, start_tol=50, end_tol=50, center_tol=100, min_fill_ratio=0.90):
    """
    Estrae un clip di 1 secondo rappresentativo da un file audio intero basandosi
    sul rilevamento del parlato tramite pydub.

    Parametri:
      - start_tol: tempo massimo in ms entro il quale il parlato deve iniziare dal bordo sinistro del clip.
      - end_tol: tempo massimo in ms dal bordo destro entro cui il parlato deve terminare.
      - center_tol: tolleranza in ms sulla posizione del centro del parlato rispetto a 500 ms.
      - min_fill_ratio: rapporto minimo tra la durata complessiva delle regioni non silenziose ed il clip (1000 ms).
    
    Restituisce un oggetto AudioSegment lungo esattamente 1000 ms se il clip è "ben riempito",
    altrimenti restituisce None.
    """
    try:
        audio = AudioSegment.from_wav(file_path)
    except Exception:
        return None

    if len(audio) < 1000:
        return None

    # Calcola la soglia di silenzio dinamicamente
    silence_thresh = audio.dBFS - 20
    # Rileva tutte le parti non silenziose
    non_silent = detect_nonsilent(audio, min_silence_len=30, silence_thresh=silence_thresh)
    if not non_silent:
        return None

    overall_start = non_silent[0][0]
    overall_end = non_silent[-1][1]
    if overall_end - overall_start < 1000:
        return None

    # Calcola il centro dell'intervallo complessivo del parlato
    candidate_center = (overall_start + overall_end) // 2
    candidate_start = candidate_center - 500
    candidate_end = candidate_center + 500

    # Assicura che il clip sia all'interno dell'audio
    if candidate_start < 0:
        candidate_start = 0
        candidate_end = candidate_start + 1000
    if candidate_end > len(audio):
        candidate_end = len(audio)
        candidate_start = candidate_end - 1000

    candidate_clip = audio[candidate_start:candidate_end]

    # Calcola nuovamente i segmenti non silenziosi nel clip candidato
    candidate_silence_thresh = candidate_clip.dBFS - 20
    candidate_non_silent = detect_nonsilent(candidate_clip, min_silence_len=30, silence_thresh=candidate_silence_thresh)
    if not candidate_non_silent:
        return None

    # Calcola il "fill ratio": quanto del clip è coperto da parlato
    total_non_silent_duration = sum([end - start for start, end in candidate_non_silent])
    fill_ratio = total_non_silent_duration / 1000.0
    if fill_ratio < min_fill_ratio:
        return None

    # Controlla che il parlato inizi abbastanza presto nel clip
    first_non_silent_start = candidate_non_silent[0][0]
    if first_non_silent_start > start_tol:
        return None

    # Controlla che il parlato termini abbastanza vicino alla fine del clip
    last_non_silent_end = candidate_non_silent[-1][1]
    if (1000 - last_non_silent_end) > end_tol:
        return None

    # Controlla che il centro del parlato sia intorno a 500ms
    candidate_spoken_center = (first_non_silent_start + last_non_silent_end) / 2.0
    if abs(candidate_spoken_center - 500) > center_tol:
        return None

    return candidate_clip

def process_audio_files(input_folder, output_folder):
    """
    Processa i file WAV nella cartella di input:
      - I file il cui nome è formato esattamente da 5 cifre vengono copiati direttamente.
      - Gli altri file vengono "vagliati" per estrarre clip di 1 secondo che siano ben riempite (speech filled);
        se il clip soddisfa i criteri, viene salvato nella cartella di output.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(".wav"):
            file_path = os.path.join(input_folder, file_name)
            base_name, _ = os.path.splitext(file_name)
            # Se il nome è formato esattamente da 5 cifre, copia direttamente
            if re.match(r"^\d{5}$", base_name):
                shutil.copy2(file_path, os.path.join(output_folder, file_name))
                continue

            candidate_clip = extract_representative_clip(file_path)
            if candidate_clip is not None:
                output_file = os.path.join(output_folder, file_name)
                candidate_clip.export(output_file, format="wav")





if __name__ == "__main__":
    # Specifica il percorso della cartella contenente i file audio interi
    input_folder = r"C:\Users\cancr\OneDrive\Desktop\train-clean-100-flat"
    # Specifica la cartella di output in cui verranno salvati i clip validi da 1 secondo
    output_folder = r"C:\Users\cancr\OneDrive\Desktop\prova"
    process_audio_files(input_folder, output_folder)
