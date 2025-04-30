import pandas as pd
import os
from tqdm import tqdm
from pydub import AudioSegment

LABELS_FILE_PATH = "PodcastFillers_Decentered.csv"
DATA_PATH_BASE = (
    r"D:\Podcast_filler_dataset\PodcastFillers\PodcastFillers\audio\episode_wav"
)
OUTPUT_FOLDER = "Filler_Clips_Decentered_Files"


def find_dataset(filename):
    for file in os.listdir(os.path.join(DATA_PATH_BASE, "validation")):
        if file == filename:
            return "validation"
    for file in os.listdir(os.path.join(DATA_PATH_BASE, "test")):
        if file == filename:
            return "test"
    return "train"


# read the labels dataframe
labels_df = pd.read_csv(LABELS_FILE_PATH)

n = len(labels_df)

# create output folders if they do not exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(os.path.join(OUTPUT_FOLDER, "train"))
    os.makedirs(os.path.join(OUTPUT_FOLDER, "test"))
    os.makedirs(os.path.join(OUTPUT_FOLDER, "validation"))
    os.makedirs(os.path.join(OUTPUT_FOLDER, "extra"))

for i, row in tqdm(labels_df.iterrows(), desc="clips"):
    # wav file name
    episode_file = (
        rf"{row['podcast_filename']}.wav".replace("?", "_")
        .replace('"', "_")
        .replace("|", "_")
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("<", "_")
        .replace(">", "_")
    )
    # dataset type (this strategy also addresses 'extra')
    dataset_type = row['episode_split_subset'] #find_dataset(episode_file)
    # clip subset, among train, test, validation, extra
    clip_subset = row["clip_split_subset"]
    # choose where to crop in the original episode file
    clip_start_s = row["clip_start_inepisode"]
    clip_end_s = row["clip_end_inepisode"]
    # full file path
    full_path = os.path.join(DATA_PATH_BASE, dataset_type, episode_file)
    # load audio segment
    audio = AudioSegment.from_file(full_path)
    # slice it according to label
    clip = audio[clip_start_s * 1000 : clip_end_s * 1000]
    # write it to disk
    output_file_path = os.path.join(
        OUTPUT_FOLDER, clip_subset, rf"clip{i}_decentered.wav"
    )
    clip.export(output_file_path, format="wav")
