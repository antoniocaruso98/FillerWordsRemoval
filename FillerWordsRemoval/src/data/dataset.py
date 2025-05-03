# filepath: FillerWordsRemoval/src/data/dataset.py
import os
import pandas as pd
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import spectrogram as sp

class AudioDataset(Dataset):
    def __init__(self, root_folder: str, type: str, transform=None):
        self.root_folder = root_folder
        self.type = type
        self.transform = transform
        self.labels_file_path = os.path.join(root_folder, f"PodcastFillers_Decentered_{type}_shuffled.csv")
        self.labels_df = pd.read_csv(self.labels_file_path)
        # list all unique classes (column: 'label') in alphabetical order
        self.classes_list = sorted(self.labels_df["label"].unique().tolist())

        # associate each class name to an integer in the range [0, n-1]
        # for fast access Class name --> Class index
        self.classes_dict = {e: i for i, e in enumerate(self.classes_list)}


    def __getitem__(self, index):
        line = self.labels_df.iloc[index]
        label_str = line["label"]
        delta_t = line["delta_t"]
        center_t = line["center_t"]
        label_nr = self.classes_dict[label_str]
        label_one_hot = torch.zeros(len(self.classes_list))
        label_one_hot[label_nr] = 1
        full_label = torch.cat((label_one_hot, torch.tensor([center_t, delta_t])))

        clip_name = self.labels_df["clip_name"].iloc[index]
        audio, sr = librosa.load(os.path.join(self.root_folder, self.type, clip_name), sr=16000)

        if self.transform:
            audio = self.transform(audio)

        spec = sp.get_db_mel_spectrogram(audio, n_fft=512, n_mels=128, sr=sr)
        spec = sp.square_spectrogram(spec, size=224)
        spec = sp.normalize_spectrogram(spec)

        return (torch.tensor(spec).unsqueeze(0).float(), full_label.float())

    def __len__(self):
        return len(self.labels_df)