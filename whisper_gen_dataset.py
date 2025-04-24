import os
import whisper
from pydub import AudioSegment

# Percorso della cartella contenente i file audio (aggiornalo se necessario)
audio_folder = os.path.join("..", "..", "..", "dev-clean-flatten")
MAX_DURATION_MS = 1000  # Durata target della clip: 1 secondo

# Carica il modello Whisper con word_timestamps abilitato
model = whisper.load_model("small")

# Itera su ogni file audio .wav nella cartella
for audio_file in os.listdir(audio_folder):
    if not audio_file.endswith(".wav"):
        continue

    file_path = os.path.join(audio_folder, audio_file)
    
    # Trascrizione con word_timestamps
    result = model.transcribe(file_path, language="en", word_timestamps=True)
    
    # Carica l'audio con pydub
    audio = AudioSegment.from_wav(file_path)
    
    # Itera su ciascun segmento trascritto
    for seg_idx, seg in enumerate(result['segments']):
        # Se non sono presenti word timestamps o il segmento è troppo corto, salta
        if 'words' not in seg or len(seg['words']) == 0:
            continue
        
        seg_start_ms = int(seg['start'] * 1000)
        seg_end_ms   = int(seg['end'] * 1000)
        if seg_end_ms - seg_start_ms < MAX_DURATION_MS:
            continue  # Segmento non abbastanza lungo
        
        # Calcola il centro del segmento per valutare la "centratura" della finestra candidata
        seg_center = (seg_start_ms + seg_end_ms) // 2

        # Estrai i confini delle parole (in millisecondi)
        words = seg['words']
        word_boundaries = [(int(w['start'] * 1000), int(w['end'] * 1000)) for w in words]

        candidate_windows = []
        n = len(word_boundaries)
        # Costruisci finestre candidate: per ogni possibile inizio (da un confine parola)
        # e per ogni possibile fine (dalla stessa o di una parola successiva) se la finestra è esattamente 1 sec
        for i in range(n):
            # Inizio candidato: il confine iniziale della parola corrente
            cand_start = word_boundaries[i][0]
            # Assicurati che non sia antecedente l'inizio del segmento
            if cand_start < seg_start_ms:
                cand_start = seg_start_ms
            for j in range(i, n):
                cand_end = word_boundaries[j][1]
                if cand_end > seg_end_ms:
                    break  # La finestra eccede il segmento
                duration = cand_end - cand_start
                if duration == MAX_DURATION_MS:
                    candidate_windows.append((cand_start, cand_end))
                elif duration > MAX_DURATION_MS:
                    break  # Non serve proseguire in quanto la durata sarà solo maggiore

        # Se non esiste nessuna finestra candidate di esattamente 1s, passa al segmento successivo
        if not candidate_windows:
            continue
        
        # Seleziona la finestra candidate il cui centro è più vicino al centro del segmento
        best_candidate = None
        best_diff = float('inf')
        for cand in candidate_windows:
            cand_center = (cand[0] + cand[1]) // 2
            diff = abs(cand_center - seg_center)
            if diff < best_diff:
                best_diff = diff
                best_candidate = cand

        if best_candidate is None:
            continue

        # Estrae la clip esattamente da cand_start a cand_end – nessun padding, nessun taglio di parola
        clip = audio[best_candidate[0]:best_candidate[1]]
        output_filename = f"{os.path.splitext(audio_file)[0]}_seg{seg_idx}_{best_candidate[0]}-{best_candidate[1]}.wav"
        clip.export(output_filename, format="wav")
        print(f"Salvato: {output_filename} (durata: {len(clip)} ms)")
