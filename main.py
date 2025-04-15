import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
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
                labels_file = 'PodcastFillers_train_labels_shuffled.csv'
            case 'test':
                labels_file = 'PodcastFillers_test_labels_shuffled.csv'
            case 'validation':
                labels_file = 'PodcastFillers_validation_labels_shuffled.csv'
        self.labels_file_path = labels_file

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
        
        # normalize the spectrogram to [0, 1]
        db_mel_spec = (db_mel_spec - np.min(db_mel_spec)) / (np.max(db_mel_spec) - np.min(db_mel_spec))

        # generate a random number between -0.5 and 0.5
        # to be used for data augmentation
        random_number =  np.random.uniform(-0.5, 0.5)
        # shift the spectrogram by a quantity proportional to the random number on time axis
        # and add the random number to the center_t value. Fill the rest of the
        # spectrogram with 0s.
        shift = int(random_number * size)
        # apply the shift to the spectrogram
        new_spectrogram = np.zeros((size, size))
        if shift > 0:
            new_spectrogram[:, shift:] = db_mel_spec[:, :-shift]
        elif shift < 0:
            new_spectrogram[:, :shift] = db_mel_spec[:, -shift:]

        db_mel_spec = new_spectrogram

        # update the center_t label
        full_label[-2] += random_number

        # plot the spectrogram for debug purposes
        
        #librosa.display.specshow(db_mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        #plt.title('MEL scale dB-spectrogram')
        #plt.colorbar(format='%+2.0f dB')
        #plt.xlabel('time (s)')
        #plt.ylabel('frequency (Hz)')
        #plt.show()
        
        
        # return (data, label)
        return (torch.tensor(db_mel_spec), full_label)
    
    def __len__(self):
        return len(self.labels_df)      


def train(model, transform, criterion, optimizer, epoch, log_interval, train_loader, device):
    """Train the model for one epoch."""
    model.train()
    losses = []
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Apply transform if provided
        if transform:
            data = transform(data)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log training stats
        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

        # Update progress bar and record loss
        pbar.update(1)
        losses.append(loss.item())

    pbar.close()
    return losses


def number_of_correct(pred, target):
    """Count the number of correct predictions."""
    return pred.eq(target).sum().item()


def get_likely_index(tensor):
    """Find the most likely label index for each element in the batch."""
    return tensor.argmax(dim=-1)


def test(model, transform, criterion, test_loader, device):
    """Evaluate the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    pbar = tqdm(total=len(test_loader), desc="Testing", leave=False)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Apply transform if provided
            if transform:
                data = transform(data)

            # Forward pass
            output = model(data)
            test_loss += criterion(output, target).item()

            # Calculate accuracy
            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)

            pbar.update(1)

    pbar.close()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.2f}%)\n")
    return test_loss, accuracy


def predictor(model, transform, device):
    """Return a prediction function bound to the model and transform."""
    def predict(tensor):
        model.eval()
        with torch.no_grad():
            tensor = tensor.to(device)
            if transform:
                tensor = transform(tensor)
            tensor = model(tensor.unsqueeze(0))
            return get_likely_index(tensor).item()
    return predict


def evaluate(losses, predict, eval_set):
    """Evaluate the model on a custom evaluation set."""
    print("\nEvaluating on validation set...")
    correct = 0

    for i, (waveform, sample_rate, utterance, *_) in enumerate(eval_set):
        try:
            output = predict(waveform)
            if output == utterance:
                correct += 1
                print('*', end="")
            else:
                print('-', end="")
        except Exception as e:
            print(f"Error during evaluation: {e}", end="")

        if (i + 1) % 100 == 0:
            print()

    accuracy = correct / len(eval_set)
    print(f"\nValidation set accuracy: {accuracy:.2f}")
    return accuracy


def initialize_model(num_classes, device):
    """Initialize the ResNet18 model for single-channel input."""
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, num_classes, bias=True)  # Adjust output layer for the number of classes
    model = model.to(device)
    return model


def prepare_dataloaders(train_set, test_set, batch_size, device):
    """Prepare DataLoader objects for training and testing."""
    num_workers = 1 if device == "cuda" else 0
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
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader



def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    root_folder = os.path.join('..', 'clip_wav')
    train_set = myDataset(root_folder, 'train')
    test_set = myDataset(root_folder, 'test')
    valid_set = myDataset(root_folder, 'validation')

    batch_size = 256
    train_loader, test_loader = prepare_dataloaders(train_set, test_set, batch_size, device)

    # Model initialization
    num_classes = len(train_set.classes_list)
    model = initialize_model(num_classes, device)
    print(model)
    summary(model, (1, 128, 128))

    # Loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=len(train_loader),
        epochs=2,
        anneal_strategy='linear'
    )

    # Training and evaluation
    n_epoch = 2
    log_interval = 20
    losses = []

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train_losses = train(model, None, criterion, optimizer, epoch, log_interval, train_loader, device)
            losses.extend(train_losses)
            test(model, None, criterion, test_loader, device)
            scheduler.step()

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Evaluate on the validation set
    predict = predictor(model, None)
    print("\nEvaluating on validation set...")
    accuracy = evaluate(losses, predict, valid_set)
    print(f"Validation set prediction accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    main()







