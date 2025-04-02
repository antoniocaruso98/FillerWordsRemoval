import wave
import audioop

def split_audio(file_path, clip_length=10, output_folder="clips"):
    # Apri il file audio
    with wave.open(file_path, 'rb') as audio_file:
        params = audio_file.getparams()
        framerate = params.framerate
        nchannels = params.nchannels
        sampwidth = params.sampwidth
        total_frames = params.nframes
        
        # Lunghezza di un clip in frame
        clip_frames = clip_length * framerate
        
        # Leggi tutti i frame
        audio_data = audio_file.readframes(total_frames)
        
        # Dividi in clip da 10 secondi
        num_clips = (total_frames + clip_frames - 1) // clip_frames  # Arrotonda verso l'alto
        
        for i in range(num_clips):
            start = i * clip_frames
            end = min((i + 1) * clip_frames, total_frames)
            clip_data = audio_data[start * sampwidth * nchannels:end * sampwidth * nchannels]
            
            # Completa il clip con silenzio se è incompleto
            if end - start < clip_frames:
                missing_frames = clip_frames - (end - start)
                silence = b'\x00' * missing_frames * sampwidth * nchannels
                clip_data += silence
            
            # Scrivi il file clip
            clip_name = f"{output_folder}/clip_{i + 1}.wav"
            with wave.open(clip_name, 'wb') as clip_file:
                clip_file.setnchannels(nchannels)
                clip_file.setsampwidth(sampwidth)
                clip_file.setframerate(framerate)
                clip_file.writeframes(clip_data)

    print(f"Audio diviso in {num_clips} clip da 10 secondi.")

# Esempio d'uso
file_path = "ytmp3free.cc_zia-tina-compilation-bella-cadduosa-e-speciali-e-youtubemp3free.org.wav"
split_audio(file_path)
