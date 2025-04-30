import pandas as pd
from tqdm import tqdm
import random

ANNOTATIONS_FILE = r'D:\Podcast_filler_dataset\PodcastFillers\PodcastFillers\metadata\PodcastFillers.csv'
#DATASET_FOLDER = r'D:\Podcast_filler_dataset\PodcastFillers\PodcastFillers\audio\episode_wav'
OUTPUT_FILE = 'PodcastFillers_Decentered.csv'
MAX_ABS_OFFSET = 0.45


fillers_df = pd.read_csv(ANNOTATIONS_FILE)

for i in range(len(fillers_df)):
    # dataset = train, test, validation, extra
    #dataset_type = row['clip_split_subset']
    # episode file name
    #filename = row['podcast_filename']

    # complete dataset path with train, test, validation, extra
    #full_dataset_path = os.path.join(DATASET_FOLDER, dataset_type)

    # offset to be added to both clip start and end
    random_offset = random.uniform(-MAX_ABS_OFFSET, MAX_ABS_OFFSET)

    # applying offset to clip position in the episode
    fillers_df.loc[i, 'clip_start_inepisode'] += random_offset
    fillers_df.loc[i, 'clip_end_inepisode'] += random_offset

    # applying reversed offset to event position inside the clip
    fillers_df.loc[i, 'event_start_inclip'] += -random_offset
    fillers_df.loc[i, 'event_end_inclip'] += -random_offset


print('Now writing output to file')
fillers_df.to_csv(OUTPUT_FILE, index=False)