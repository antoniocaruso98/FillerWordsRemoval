import librosa
import soundfile as sf
import os
import numpy as np

def split_audio_librosa(file_path, clip_length=10, output_folder="clips"):
        # docstring di info che chiamo con help(split_audio_librosa)
    '''
    Divide un file audio di formato e lunghezza arbitrari
    in una serie di clip di lunghezza specifiata in ms.
    Es: clip_length = 10s = 10 * 1000 ms.
    Se ultima clip è minore della lunghezza desiderata, padding con silenzio.

    Args:
        file_path (str): Il percorso del file audio di input.
        clip_length (int): La lunghezza desiderata di ogni clip in secondi. Default: 10.
        output_folder (str): Il percorso dove vengono salvate le clip generate. Default: "clips"
    Returns:
        None: Le clip vengono salvate come file separati nel formato mp3.

    Da modificare l'organizzazione in sottocartelle delle clip di ogni audio.
    '''

    if not os.path.exists(file_path):
        print(f"Errore: Il file '{file_path}' non esiste.")
        return
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # Load the audio file with librosa
        y, sr = librosa.load(file_path, sr=None)  # Keep original sample rate

        # Calculate clip length in samples
        clip_samples = clip_length * sr

        # Calculate the number of clips
        num_clips = int(np.ceil(len(y) / clip_samples))

        for i in range(num_clips):
            start_sample = int(i * clip_samples)
            end_sample = int(min((i + 1) * clip_samples, len(y)))

            clip = y[start_sample:end_sample]

            # Pad with silence if necessary
            if len(clip) < clip_samples:
                clip = np.pad(clip, (0, int(clip_samples - len(clip))), mode='constant')

            # Output file name
            output_path = os.path.join(output_folder, f"clip_{i + 1}.wav")

            # Save the clip using soundfile
            sf.write(output_path, clip, sr)

        print(f"Audio divided into {num_clips} clips of {clip_length} seconds.")

    except Exception as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    file_path = "ytmp3free.cc_zia-tina-compilation-bella-cadduosa-e-speciali-e-youtubemp3free.org.mp3"
    split_audio_librosa(file_path)