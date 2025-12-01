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
import spectrogram as sp
from sklearn.metrics import classification_report, confusion_matrix


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
        self.labels_file_path = os.path.join(root_folder, labels_file)

        # full path to data folder
        self.data_folder = os.path.join(root_folder, type)

        # read labels file inside a Pandas Dataframe
        self.labels_df = pd.read_csv(self.labels_file_path)

        # Discard all 'Nonfiller' clips
        self.labels_df = self.labels_df[self.labels_df["label"] != "Nonfiller"]

        # Keep only ''clip_name', 'delta_t' and 'center_t'
        self.labels_df = self.labels_df[["clip_name", "delta_t", "center_t"]]

    def __getitem__(self, index):

        # read row 'index' from the Dataframe
        line = self.labels_df.iloc[index]
        # extract clip_name
        clip_name = line["clip_name"]
        # extract delta_t
        delta_t = line["delta_t"]
        # extract center_t
        center_t = line["center_t"]

        full_label = torch.tensor([center_t, delta_t])

        # read audio file
        audio, sr = librosa.load(os.path.join(self.data_folder, clip_name), sr=16000)

        if sr != 16000:
            print(f"WARNING: sr should be 16000, but instead got {sr}")

        # Perform data augmentation
        if np.random.rand() > 0.5:  # Randomly scale volume
            gain = np.random.uniform(0.8, 1.2)  # Scale volume by 0.8x to 1.2x
            audio = audio * gain

        if np.random.rand() > 0.5:  # Randomly apply pitch shifting
            n_steps = np.random.uniform(-2, 2)  # Shift pitch by -2 to +2 semitones
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

        if np.random.rand() > 0.5:  # Randomly apply time stretching
            rate = np.random.uniform(0.8, 1.2)  # Stretch by 0.8x to 1.2x
            audio = librosa.effects.time_stretch(audio, rate=rate)
            full_label[-2] = full_label[-2] * rate
            full_label[-1] = full_label[-1] * rate

        if np.random.rand() > 0.5:  # Randomly add noise
            noise = np.random.normal(0, 0.005, audio.shape)
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

        # return (data, label)
        # note that the network expects a 3-dimensional tensor for a single sample
        # because it is intended to allow multi-channel input images. However, a
        # spectrogram only has a single channel, so a fictitious channel dimension is
        # added at position 0.
        # A 4-dimensional tensor is instead used to represent a batch of data.
        return (torch.tensor(spec).unsqueeze(0).float(), full_label.float())

    def __len__(self):
        return len(self.labels_df)


def train(
    model, criterion, optimizer, epoch, log_interval, train_loader, device, scheduler
):
    """Train the model for one epoch."""
    model.train()
    losses = []
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate scheduler
        scheduler.step()

        # Log training stats
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

        # Update progress bar and record loss
        pbar.update(1)
        losses.append(loss.item())

    pbar.close()
    return losses


def number_of_correct(output, target, iou_threshold=0.5):
    """
    Count the number of correct predictions, i.e. the number of predicted
    bounding box coordinates for which the intersection-over-union index
    is greater than 'iou_threshold'
    """

    iou = intersection_over_union(output, target)
    nr_correct = (iou >= iou_threshold).int().sum().item()
    return nr_correct


def intersection_over_union(output_bb, target_bb):
    """
    return a tensor containing iou between output and target
    bounding boxes for each sample.
    """
    # convert from (center_t, delta_t) to (start, end)
    output_se = torch.zeros_like(output_bb)
    target_se = torch.zeros_like(target_bb)

    output_se[:, 0] = output_bb[:, 0] - output_bb[:, 1] / 2
    output_se[:, 1] = output_bb[:, 0] + output_bb[:, 1] / 2

    target_se[:, 0] = target_bb[:, 0] - target_bb[:, 1] / 2
    target_se[:, 1] = target_bb[:, 0] + target_bb[:, 1] / 2

    # For each sample, compute intersection (col 0) and union (col 1)
    result = torch.zeros_like(output_bb)

    min_start = torch.min(output_se[:, 0], target_se[:, 0])
    max_start = torch.max(output_se[:, 0], target_se[:, 0])
    min_end = torch.min(output_se[:, 1], target_se[:, 1])
    max_end = torch.max(output_se[:, 1], target_se[:, 1])

    zeros = torch.zeros_like(result[:, 0])

    result[:, 0] = torch.max(zeros, min_end - max_start)
    result[:, 1] = max_end - min_start

    return result[:, 0] / result[:, 1]


def evaluate(model, criterion, loader, device, iou_threshold):
    """Evaluate the model on the test/validation set with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    pbar = tqdm(total=len(loader), desc="Testing", leave=False)

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            total_loss += criterion(output, target).item()

            # Use number_of_correct to calculate correct predictions
            correct += number_of_correct(output, target, iou_threshold)

            pbar.update(1)

    pbar.close()
    total_loss /= len(loader.dataset)

    # Calculate accuracy
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"\nAccuracy: {accuracy:.2f}%")

    return total_loss, accuracy


def initialize_model(architecture, device):
    """
    Initialize a model with the specified architecture in order to perform
    regression to compute 2 bounding box coordinates.
    """
    if architecture == "ResNet":
        # Load ResNet18
        model = torchvision.models.resnet18(weights=None)

        # Modify the first convolutional layer to accept 1 input channel
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Modify the fully connected layer to output 2 bounding box coordinates
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),  # Add a hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),  # Output: 2 continuous bounding box coordinates
        )
        print("Using ResNet18...\n")
    elif architecture == "MobileNet":
        # Load MobileNetV2
        model = torchvision.models.mobilenet_v2(weights=None)

        # Modify the first convolutional layer to accept 1 input channel
        model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        # Modify the fully connected layer to output 2 bounding box coordinates
        model.classifier = nn.Sequential(
            nn.Linear(model.last_channel, 256),  # Add a hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),  # Output: 2 continuous bounding box coordinates
        )
        print("Using MobileNetv2...\n")

    else:
        print("Unsupported architecture. Choose 'ResNet' or 'MobileNet'.")
    # Move to device
    model = model.to(device)
    return model


def prepare_dataloaders(train_set, test_set, validation_set, batch_size, device):
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
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, validation_loader


def main():
    # Starting with always the same seed for weights reproducibility
    torch.manual_seed(42)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    root_folder = os.path.join("..", "DATASET_COMPLETO_V2")
    print(f"training on {root_folder}...\n")

    # Read datasets
    train_set = myDataset(root_folder, "train")
    test_set = myDataset(root_folder, "test")
    validation_set = myDataset(root_folder, "validation")

    # Set batch size
    batch_size = 64

    # Get dataloaders
    train_loader, test_loader, validation_loader = prepare_dataloaders(
        train_set, test_set, validation_set, batch_size, device
    )

    # Model initialization
    architecture = "MobileNet"  # Choose between "ResNet" and "MobileNet"

    model = initialize_model(architecture, device)

    # Load pre-trained model if available
    best_model_path = "best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}\n")
    else:
        print("No pre-trained model found. Starting training from scratch.\n")

    # print(model)
    summary(model, (1, 224, 224))

    # Learning rate
    lr = 0.001

    # Nr. epochs
    n_epochs = 6

    # Loss function, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        anneal_strategy="linear",
    )

    # Training and evaluation
    log_interval = train_set.__len__() // (batch_size * 100)  # stamp every 1%
    losses = []

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    best_validation_loss = float("inf")
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(1, n_epochs + 1):

            # training on that epoch
            train_losses = train(
                model,
                criterion,
                optimizer,
                epoch,
                log_interval,
                train_loader,
                device,
                scheduler,
            )
            # compute avg loss for entire epoch and add to list 'losses'
            losses.append(float(np.mean(train_losses)))

            # Validation on that epoch
            validation_loss, validation_accuracy = evaluate(
                model,
                criterion,
                validation_loader,
                device,
                iou_threshold=0.5,
            )
            print(
                f"Validation set: Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.2f}%"
            )
            # Saving the best model
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), "best_model.pth")

    # Final evaluation on the test set
    test_loss, test_accuracy = evaluate(
        model, criterion, test_loader, device, iou_threshold=0.5
    )
    print(f"Test set: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
