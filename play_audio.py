import sounddevice as sd
import librosa
import os
from main import myDataset


def play_audio(audio, sr):
    """Riproduce un campione audio."""
    print("Riproduzione audio in corso...")
    sd.play(audio, samplerate=sr)
    sd.wait()  # Aspetta che la riproduzione termini
    print("Riproduzione terminata.")


root_folder = os.path.join("..", "Dataset_completo")
test_set = myDataset(root_folder, "test")

# Carica il campione
index = 1000  # Cambia l'indice per selezionare un campione diverso
spec, label = test_set[index]
clip_name = test_set.labels_df["clip_name"].iloc[index]
audio, sr = librosa.load(os.path.join(test_set.data_folder, clip_name), sr=16000)

# Riproduci l'audio
play_audio(audio, sr)
# lo spettrogramma viene visualizzato se abilitato nel getitem()