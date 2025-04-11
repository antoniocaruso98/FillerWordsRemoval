import torch
import os
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
        
        # creating LOG-MEL spectrogram
        
        # define window length = nr. samples in time/frequency
        n_fft = 512
        win_length = n_fft
        # define hop length
        hop_length = win_length // 2
        # define n_mels = nr. frequency bins
        n_mels = 128
        # define maximum frequency = 0.5 * sampling rate
        fmax = 0.5 * sr
        
        # generate spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=fmax, n_fft=n_fft,
                                          win_length=win_length, hop_length=hop_length)
        
        # convert to Power Spectrogram by taking modulus and squaring
        mel_spec = np.abs(mel_spec) ** 2
        # convert power to dB
        db_mel_spec = librosa.power_to_db(mel_spec)

        # the spectrogram must now be converted to a square image (size x size)
        size = 224
        db_mel_spec = cv2.resize(db_mel_spec, (size, size), interpolation=cv2.INTER_NEAREST)

        # plot the spectrogram for debug purposes
        '''
        librosa.display.specshow(db_mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.title('MEL scale dB-spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel('time (s)')
        plt.ylabel('frequency (Hz)')
        plt.show()
        '''
        
        # return (data, label)
        return (torch.tensor(db_mel_spec), full_label)
    
    def __len__(self):
        return len(self.labels_df)      


def train(model, transform, criterion, optimizer, epoch, log_interval, train_loader, device):
    # set model in training mode
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        if transform != None :
          data = transform(data)
        output = model(data)

        # negative log-likelihood for a tensor of size (batch x 1 x n_output)
        # loss = F.nll_loss(output.squeeze(), target)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if batch_idx % log_interval == 0:
            print()
            print(f"       Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]")
            print(f"       Loss: {loss.item():.6f}")

        # update progress bar
        pbar.update(pbar_update)

        # record loss
        losses.append(loss.item())        






def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def test(model, transform, criterion, epoch):
    model.eval()
    correct = 0
    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        if transform != None :
          data = transform(data)

        output = model(data)
        loss = criterion(output, target)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        # update progress bar
        pbar.update(pbar_update)


    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")        




# Returns the predict function binded to the model and transform
def predictor(model, transform):

    def p(tensor):
      # Use the model to predict the label of the waveform
      tensor = tensor.to(device)
      tensor = transform(tensor)
      tensor = model(tensor.unsqueeze(0))
      tensor = get_likely_index(tensor)
      tensor = index_to_label(tensor.squeeze())
      return tensor

    return p




def evaluate(losses, predict, eval_set):

  # Let's plot the training loss versus the number of iteration.
  plt.plot(losses);
  plt.title("training loss");

  cnt = 0

  for i, (waveform, sample_rate, utterance, *_) in enumerate(eval_set):
      try:
          output = predict(waveform)
      except:
          None
          # print("An exception occurred ", utterance, output)
      if output != utterance:
          # ipd.Audio(waveform.numpy(), rate=sample_rate)
          # print(f"Data point #{i}. Expected: {utterance}. Predicted: {output}.")
          print('-', end="")
      else:
          print('*', end="")
          cnt = cnt + 1
      if(not((i+1) % 100)):
        print()

  return cnt/len(eval_set)





if __name__ == '__main__':
    os.system('cls')
    ds = myDataset('clip_wav', 'train')
    
    #audio = librosa.load('clip_wav/train/00001.wav')
    audio, label = ds[0]

    #print(audio.shape)
    #print(label)
    sd.play(audio, samplerate=16000)
    sd.wait()