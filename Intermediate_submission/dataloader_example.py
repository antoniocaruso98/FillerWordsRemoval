import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import pandas as pd
import os
import numpy as np
import spectrogram as sp

class myDataset(Dataset):

    def __init__(self, root_folder: str, type: str):
        # set root folder for dataset
        self.root_folder = root_folder
        # check if train, test or validation set and
        # select appropriate annotations file
        if type not in ["train", "test", "validation"]:
            type = "train"

        self.type = type
        match self.type:
            case "train":
                labels_file = "PodcastFillers_Decentered_train_shuffled.csv"
            case "test":
                labels_file = "PodcastFillers_Decentered_test_shuffled.csv"
            case "validation":
                labels_file = "PodcastFillers_Decentered_validation_shuffled.csv"
        self.labels_file_path = os.path.join(root_folder,labels_file)

        # full path to data folder
        self.data_folder = os.path.join(root_folder, type)

        # read labels file inside a Pandas Dataframe
        self.labels_df = pd.read_csv(self.labels_file_path)
        # list all unique classes (column: 'label') in alphabetical order
        self.classes_list = sorted(self.labels_df["label"].unique().tolist())

        # associate each class name to an integer in the range [0, n-1]
        # for fast access Class name --> Class index
        self.classes_dict = {e: i for i, e in enumerate(self.classes_list)}


    def __getitem__(self, index):
        # read row 'index' from the Dataframe
        line = self.labels_df.iloc[index]
        # extract raw string label
        label_str = line["label"]
        # extract delta_t
        delta_t = line["delta_t"]
        # extract center_t
        center_t = line["center_t"]
        # convert raw string to number
        label_nr = self.classes_dict[label_str]
        # convert label using one-hot encoding
        label_one_hot = torch.zeros(len(self.classes_list))
        label_one_hot[label_nr] = 1
        # create torch tensor: [...one_hot_label..., delta_t, center_t]
        full_label = torch.cat((label_one_hot, torch.tensor([delta_t, center_t])))

        # now reading actual data from file
        clip_name = self.labels_df["clip_name"].iloc[index]
        audio, sr = librosa.load(os.path.join(self.data_folder, clip_name), sr=16000)

        # Apply data augmentation only for training set
        if self.type == "train":
            if np.random.rand() > 0.5:  # Randomly scale volume
                gain = np.random.uniform(0.9, 1.1)  # Scale volume by 0.9x to 1.1x
                audio = audio * gain
                
            if np.random.rand() > 0.5:  # Randomly apply pitch shifting
                n_steps = np.random.uniform(-1, 1)  # Shift pitch by -1 to +1 semitones
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

            if np.random.rand() > 0.5:  # Randomly apply time stretching
                rate = np.random.uniform(0.9, 1.1)  # Stretch by 0.9x to 1.1x
                audio = librosa.effects.time_stretch(audio, rate=rate)
                full_label[-2] = full_label[-2] * rate
                full_label[-1] = full_label[-1] * rate

            if np.random.rand() > 0.5:  # Randomly add noise
                noise = np.random.normal(0, 0.004, audio.shape)
                audio = audio + noise

        # creating LOG-MEL spectrogram
        n_fft = 512
        n_mels = 128
        spec = sp.get_db_mel_spectrogram(audio, n_fft=n_fft, n_mels=n_mels, sr=sr)

        # converting spectrogram to a square image (size x size)
        size = 224
        spec = sp.square_spectrogram(spec, size)

        # normalize the spectrogram values to [0, 1]
        spec = sp.normalize_spectrogram(spec)

        # plot the spectrogram for debug purposes
        #sp.plot_spectrogram(spec, sr, hop_length=(n_fft // 2))

        # return (data, label)
        return (torch.tensor(spec).unsqueeze(0).float(), full_label.float())

    def __len__(self):
        return len(self.labels_df)

def prepare_dataloaders(train_set, test_set, validation_set, batch_size, device, num_workers=8):
    """Prepare DataLoader objects for training and testing."""
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers//2,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, validation_loader

if __name__ == '__main__':
    pass
    #ds = myDataset()