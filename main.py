import torch
import os
import pandas as pd
import librosa
import sounddevice as sd
from torch.utils.data import Dataset, DataLoader

class myDataset (Dataset):
    
    def __init__(self, root_folder: str, type: str):
        # set root folder for dataset
        self.root_folder = root_folder
        # check if train, test or validation set and
        # select appropriate annotations file
        if type not in ['train', 'test', 'validation']:
            type = 'train'
        
        self.type = type
        match self.type:
            case 'train':
                labels_file = 'PodcastFillers_train_labels.csv'
            case 'test':
                labels_file = 'PodcastFillers_test_labels.csv'
            case 'validation':
                labels_file = 'PodcastFillers_validation_labels.csv'
        self.labels_file_path = os.path.join(self.root_folder, labels_file)

        # full path to data folder
        self.data_folder = os.path.join(root_folder, type)
        
        # read labels file inside a Pandas Dataframe
        self.labels_df = pd.read_csv(self.labels_file_path)
        # list all unique classes (column: 'label')
        self.classes_list = self.labels_df['label'].unique().tolist()

        # associate each class name to an integer in the range [0, n-1]
        # for fast access Class name --> Class index
        self.classes_dict = {}
        for (i, e) in enumerate(self.classes_list):
            self.classes_dict.update({e : i})
    
    def __getitem__(self, index):
        # read row 'index' from the Dataframe
        line = self.labels_df.iloc[index]
        # extract raw string label
        label_str = line['label']
        # extract delta_t
        delta_t = line['delta_t']
        # extract center_t
        center_t = line['center_t']
        # convert raw string to number
        label_nr = self.classes_dict[label_str]
        # convert label using one-hot encoding
        label_one_hot = torch.zeros(len(self.classes_list))
        label_one_hot[label_nr] = 1
        # create torch tensor: [...one_hot_label..., center_t, delta_t]
        full_label = torch.cat((label_one_hot, torch.tensor([center_t, delta_t])))

        # now reading actual data from file
        clip_name = self.labels_df['clip_name'].iloc[index]
        audio, sr = librosa.load(os.path.join(self.data_folder, clip_name), sr=16000)
        #print(f'sampling rate = {sr}')
        audio_clip = torch.tensor(audio)
        
        # return (data, label)
        return (audio_clip, full_label)
    
    def __len__(self):
        return len(self.labels_df)

        

if __name__ == '__main__':
    os.system('cls')
    ds = myDataset('clip_wav', 'train')
    
    #audio = librosa.load('clip_wav/train/00001.wav')
    audio, label = ds[0]

    #print(audio.shape)
    #print(label)
    sd.play(audio, samplerate=16000)
    sd.wait()