import argparse
import sys
import os




def intersect_intervals(intervals):
    n = len(intervals)

    # Ordina gli intervalli per inizio
    intervals.sort()
    

    input_list= []
    output_list= intervals
    current_lenght= n
    prev_lenght= 0
    while current_lenght!= prev_lenght:
        prev_lenght= current_lenght
        input_list= output_list
        output_list= []
        print("ho fatto un giro")
        i = 0
        while i < len(input_list):
            a1, b1 = input_list[i]
            if i!=len(input_list)-1:
                a2, b2 = input_list[i+1]
                # Verifica sovrapposizione
                if a2 <= b1 and b2 >= a1:
                    # Calcola l'unione
                    start = min(a1, a2)
                    end = max(b1, b2)
                    output_list.append((start, end))
                    i+= 1
                else:
                    output_list.append((a1, b1))
            else:
                output_list.append((a1, b1))
            i += 1
        output_list.sort()
        current_lenght= len(output_list)


    return output_list




# Raggruppa gli indici continui di silenzio in intervalli
def find_silent_intervals(power, sr, threshold, min_duration_s):
    """
    Identify all sufficiently large intervals in the signal (at least **min_duration_s** seconds)
    in which the signal **power** is lower than the chosen **threshold**. Returns a list
    containing tuples of type (start, end), each representing an interval (start and end are
    expressed in seconds).
    """

    # Select only indices for which power < threshold
    silent_indices = np.where(power < threshold)[0]

    # Use the 'times' array to convert between power array indexes and seconds:
    # times[<power_index>] = <corresponding time in seconds>
    times = times_like(power, sr=sr)

    # If there is no silent interval, return empty list
    if len(silent_indices) == 0:
        print("Audio does not contain any silent intervals")
        return []

    # Initialize empty list
    intervals = []

    # Read first start index
    start = silent_indices[0]

    for i in range(1, len(silent_indices)):
        # If current index is not contiguous to the previous one, this
        # means that the previous index is the end of an interval
        if silent_indices[i] != (silent_indices[i - 1] + 1):
            # Found an interval -> check duration and in case add to list
            if times[silent_indices[i - 1]] - times[start] >= min_duration_s:
                intervals.append(
                    (float(times[start]), float(times[silent_indices[i - 1]]))
                )
            # New interval starts at current index
            start = silent_indices[i]

    # Last interval ends because of array ending (other intervals end
    # because the following index in the array is not contiguous)
    if times[silent_indices[-1]] - times[start] >= min_duration_s:
        intervals.append((float((times[start])), float(times[silent_indices[-1]])))

    return intervals


def remove_silence(audio_path):
    """
    Given the path to an audio file, **audio_path**, replace all sufficiently
    long silent intervals, with shorter ones, thus reducing audio length.
    Returns a tuple **(output_audio, sr)**, where sr is the actual sampling
    frequency.
    """
    # Load audio file
    audio, sr = load(audio_path, sr=16000)
    print(f"Correctly loaded audio from: {audio_path}")
    print(f"Duration: {len(audio)/sr:.2f} seconds, Sampling Rate: {sr} Hz")

    # Compute audio signal instant power
    power = rms(y=audio).flatten()

    # Define a power threshold to distinguish between sound and silence
    threshold = float(min(power)) + 0.1 * (float(max(power)) - float(min(power)))

    # Define a minimum duration (expressed in seconds) which silent intervals
    # must have in order to be considered
    min_duration_s = 0.150

    # Identify contiguous sets of samples which correspond to silence
    silent_intervals = find_silent_intervals(power, sr, threshold, min_duration_s)

    if debug:
        # Print interval with maximum length, tot. nr. of silence seconds, nr. silence intervals
        if silent_intervals:
            max_ind = -1
            max_len = -1
            tot_silence_s = 0
            for i, (start, end) in enumerate(silent_intervals):
                tot_silence_s += end - start
                if end - start > max_len:
                    max_ind = i
                    max_len = end - start
            print(
                "Max length interval : ",
                f"{float(silent_intervals[max_ind][0])//60}m {float(silent_intervals[max_ind][0])%60}s",
                " - ",
                f"{float(silent_intervals[max_ind][1])//60}m {float(silent_intervals[max_ind][1])%60}s",
            )
            print(f"Tot seconds of silence: {tot_silence_s}")
            print(f"Nr. of silence intervals: {len(silent_intervals)}")

    # Output a new audio (librosa format) by substituting silent intervals with
    # short intervals filled with 0s with length = min_duration
    output_audio = []
    last_end_sample = 0
    for start, end in silent_intervals:
        output_audio.append(audio[int(last_end_sample * sr) : int(start * sr)])
        # Add small silent interval
        output_audio.append(audio[int(start * sr): int(start * sr)+int(min_duration_s * sr)])
        last_end_sample = end

    # Add the remaining part of the audio. No silent interval can be found
    # here, since last_end_sample is the end of the last silent interval
    output_audio.append(audio[int(last_end_sample * sr) :])

    # Concatenate all fragments and generate output audio
    output_audio = np.concatenate(output_audio)

    if debug:
        # Save output audio as file
        sf.write("audio_clean.wav", output_audio, sr)
        print("Exported clean audio file")

    return output_audio, sr


def load_batch(audio, sr, batch_size, batch_nr, stride):
    """
    Try to build a batch containing **batch_size** dB MEL-spectrograms from audio,
    starting from the offset corresponding to **batch_nr**. If the **audio** signal
    is not long enough, then the returned **batch** has a shorter len than **batch_size**
    and only contains full clips.
    """
    # Length (in samples) of 1 second clip
    clip_length = sr

    # Offset (expressed in nr. of samples) of the batch start in the audio array
    batch_offset = int( batch_nr * (clip_length + (batch_size-1)*stride*clip_length) )

    # Build the batch, starting from batch offset. The loop tries
    # to extract 'batch_size' clips and, if this is not possibile due to the
    # the audio array reaching the end, then the batch is built using
    # only the full extracted clips -> The last, incomplete clip is discarded.
    clips = []
    for clip_nr in range(batch_size):
        start_sample = batch_offset + int(clip_nr * clip_length * stride)
        end_sample = start_sample + clip_length
        if end_sample <= len(audio):
            clips.append(audio[start_sample:end_sample])
        else:
            break

    # Generate spectrograms. Add an 'artificial' dimension in position 0,
    # since the model expects a three-dimensional tensor for each sample,
    # where the first dimension is the channel number.
    spectrograms = []
    for clip in clips:
        s = sp.get_db_mel_spectrogram(clip, 512, 128, sr)
        s = sp.square_spectrogram(s, 224)
        s = sp.normalize_spectrogram(s)
        spectrograms.append(torch.tensor(s).unsqueeze(0).float())

    # Compose the batch by stacking the three-dimensional spectrograms
    # Along a fourth dimension added in position 0, so that the batch
    # can be correctly processed by the network.
    batch = torch.stack(spectrograms, dim=0)

    return batch


def load_model(architecture, weights_file_path):
    """
    Load the model specified by the **architecture** parameter, which must
    be one between 'ResNet' and 'MobileNet', using the weights file pointed
    by **weights_file_path**.
    """
    # set number of classes
    nr_classes = 7
    # choose between GPU (if available) and CPU
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device= "cpu"

    # Load model and weights
    model = initialize_model(architecture, nr_classes, device)
    checkpoint = torch.load(weights_file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Remove filler words from an input wav file."
    )

    # Defining arguments
    parser.add_argument("-f", "--file", type=str, help="Input audio file path")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug mode (print additional info)",
    )
    parser.add_argument("-w", "--weights", type=str, help="Weights file path")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Network model to be used",
        choices=["ResNet", "MobileNet"],
        default="ResNet",
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output audio file path", default="output.wav"
    )

    # Parsing degli argomenti
    args = parser.parse_args()

    # check arguments

    if args.file is None or args.weights is None:
        print("Invalid arguments")
        sys.exit(1)

    return args.file, args.debug, args.weights, args.output


def main():

    #stride
    #stride= 0.25
    stride=1
    #stride= 0.5
    #stride= 0.2
    # Set maximum size of batches to be processed by the model
    batch_size = 64
    # List of all recognizable classes
    classes_list = ["Breath", "Laughter", "Music", "Nonfiller", "Uh", "Um", "Words"]
    # Class index for negative class, which does not require localization
    negative_class_index = classes_list.index("Nonfiller")
    # Nr. of classes
    nr_classes = len(classes_list)
    # Clip len in samples is equal to sr, since clips have 1 second length
    clip_length = sr
    # Initialize the list containing all audio frames to be removed as
    # (start, end) tuples, where start and end are expressed in samples and are
    # relative to the clean_audio array
    audio_fragments_to_be_removed = []

    # List containing log of events in the format (start_s, end_s, filler_class)
    events_log = []

    # Iterate on all batches in the 'clean_audio' signal and send them as input
    # to the model to get predictions. End when the batch has a length which is
    # less than the maximum, i.e. 'batch_size'.
    batch_nr = 0
    end = False
    while not end:
        # Load a batch
        batch = load_batch(clean_audio, sr, batch_size, batch_nr, stride)
        # Check if it is the last one
        if len(batch) < batch_size:
            end = True
        # Compute offset of the batch in the audio array (in samples)
        batch_offset = int( batch_nr * (clip_length + (batch_size-1)*stride*clip_length) )
        # Get prediction
        output = model(batch)

        # (Delta, center)
        bb= output[:, nr_classes:]
        # Convert to (init, end) in samples
        output_xmin = ( (bb[:, 1] - (bb[:, 0] / 2)) * sr).int()
        output_xmax = ( (bb[:, 1] + (bb[:, 0] / 2)) * sr).int()
        event_len= output_xmax - output_xmin

        # class labels do not need any transformations, since the predicted class
        # is simply the one having the maximum predicted value.
        labels = output[:, :nr_classes]

        # For each clip in the output
        for i in range(len(output)):
            # read class index from label (which is one-hot encoded)
            output_class_index = labels[i].argmax().item()
            # If negative class, do nothing for this clip
            if output_class_index == negative_class_index:
                continue
            # If the class is not negative, then we need to remove the corresponding

            # Compute clip offset in batch (in samples)
            clip_offset_in_batch = i * clip_length * stride
            # Compute event inclip start offset (in samples)
            event_offset_inclip = output_xmin[i]
            # Compute offset (samples) in the audio array corresponding to event start
            event_offset_in_audio = (
                batch_offset + clip_offset_in_batch + event_offset_inclip
            )
            # Compute start and end of event (in samples)
            event_start_in_audio = event_offset_in_audio
            event_end_in_audio = event_start_in_audio + event_len[i]

           # Neglecting words
            if output_class_index == classes_list.index("Words"): 
                continue
             # Append this interval to the list containing the fragments to be removed
            audio_fragments_to_be_removed.append(
                (event_start_in_audio.int().item(), event_end_in_audio.int().item())
            )
            events_log.append(
                (
                    (event_offset_in_audio / sr).item(),
                    (event_end_in_audio / sr).item(),
                    classes_list[output_class_index],
                )
            )

        # Continue with next batch
        batch_nr += 1

    # Now remove fragments containing fillers
    final_audio_list = []
    filler_silence_s = 0.100
    last_end_sample = 0
    print(f"Nr. of fragments to be removed: {len(audio_fragments_to_be_removed)}")
    audio_fragments_to_be_removed = intersect_intervals(audio_fragments_to_be_removed)
    print(f"Nr. of fragments to be removed after intersection: {len(audio_fragments_to_be_removed)}")

    [print(s/sr,e/sr) for s,e in audio_fragments_to_be_removed]

    # fade in/out
    fade_duration = 0.150 #s 
    # max/min dynamic duration
    fade_min, fade_max = 0.08, 0.180 
    silenzio_min, silenzio_max = 0.30, 1.00
    min_filler = 0.150
    max_filler = 0.600 *2

    for start, end in audio_fragments_to_be_removed:
        # print(f"start = {start}, end = {end}")
        if start < 0:
            start = 0
        if end > len(clean_audio):
            end = len(clean_audio)
        
        # Do not consider filler too short
        if (end-start)/sr < min_filler:
            final_audio_list.append(clean_audio[last_end_sample:end])
            last_end_sample = end
            continue

        # durata_filler in s
        durata_filler = (end - start)/sr
        # Scaling values
        scaling = (durata_filler-min_filler)/(max_filler-min_filler)
        scaling= min(max(scaling, 0), 1)
        # durations
        fade_duration = int(fade_min + (fade_max - fade_min) * (scaling))
        filler_silence_s = int(silenzio_min + (silenzio_max - silenzio_min) * scaling)

        # Segmento precedente con fade-out
        clean_audio[start-int(fade_duration*sr):start] = apply_fade(clean_audio[start-int(fade_duration*sr):start], fade_duration, sr, fade_type="out")
        final_audio_list.append(clean_audio[last_end_sample : start])

        # Silence
        final_audio_list.append(np.zeros(int(filler_silence_s * sr)))

        # Segmento successivo con fade-in
        next_segment = clean_audio[end:end+int(fade_duration*sr)]
        next_segment = apply_fade(next_segment, fade_duration, sr, fade_type="in")
        #final_audio_list.append(next_segment)
    
        last_end_sample = end
    final_audio_list.append(clean_audio[last_end_sample:])

    final_audio = np.concatenate(final_audio_list)

    return final_audio


def apply_fade(audio_segment, fade_duration, sr, fade_type="in"):
    """Applica un fade-in o fade-out a un segmento audio."""
    fade_samples = int(fade_duration * sr)
    
    # Se il segmento è più corto del fade, riduciamo la durata del fade
    if len(audio_segment) < fade_samples:
        fade_samples = len(audio_segment)

    # non-linear fade curve (human perception)
    fade_curve = np.sin(np.linspace(0, np.pi/2, fade_samples)) if fade_type == "in" else np.sin(np.linspace(np.pi/2, 0, fade_samples))
    
    if fade_type == "in":
        audio_segment[:fade_samples] *= fade_curve
    else:
        audio_segment[-fade_samples:] *= fade_curve
    
    return audio_segment



if __name__ == "__main__":

    # Parse command line arguments
    audio_path, debug, weights_path, output_path = parse_args()

    # Load external modules
    from librosa import load, times_like
    from librosa.feature import rms
    import numpy as np
    import soundfile as sf
    import torch
    import spectrogram as sp
    from main import initialize_model

    # Load audio file and perform pre-processing by removing long silent
    # intervals and substituting them with short ones of certain length.
    clean_audio, sr = remove_silence(audio_path)

    # Load model
    model = load_model("ResNet", os.path.join("..","checkpoint.pth"))


    final_audio = main()

    sf.write(output_path, final_audio, sr)
    print("AUDIO SALVATO!")


#python .\inference.py -f .\ytmp3free.cc_zia-tina-compilation-bella-cadduosa-e-speciali-e-youtubemp3free.org.mp3 -w funci -m ResNet -o output_prova_zia_tina.wav