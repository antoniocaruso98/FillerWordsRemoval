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
        # create torch tensor: [...one_hot_label..., center_t, delta_t]
        full_label = torch.cat((label_one_hot, torch.tensor([center_t, delta_t])))

        # now reading actual data from file
        clip_name = self.labels_df["clip_name"].iloc[index]
        audio, sr = librosa.load(os.path.join(self.data_folder, clip_name), sr=16000)

        # Apply data augmentation
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

        
        '''# shift spectrogram on time axis by random amount (+- half size).
        # The remaining part is filled with noise.
        if 'rate' in locals(): # if it is present the time stretching, we modulate the shift
            random_number = np.random.uniform(-0.5 * rate, 0.5 * rate)
        else:
            random_number = np.random.uniform(-0.5, 0.5)
        shift = int(random_number * size)
        noise_level= sp.calculate_noise_level(audio, snr_db=30) # Mid-level noise
        spec = sp.shift_spectrogram(spec, shift, noise_level)

        # update the center_t label
        full_label[-2] += random_number'''

        # plot the spectrogram for debug purposes
        #sp.plot_spectrogram(spec, sr, hop_length=(n_fft // 2))

        # return (data, label)
        return (torch.tensor(spec).unsqueeze(0).float(), full_label.float())

    def __len__(self):
        return len(self.labels_df)


def train(model, criterion, optimizer, epoch, log_interval, train_loader, device, scheduler):
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


def number_of_correct(output, target, iou_threshold,negative_class_index):
    """Count the number of correct predictions."""

    # discard last two columns in 'output' and 'target'
    # because they do not represent classes but bounding
    # box coordinates.
    predicted_class = get_likely_index(output[:, 0:-2])
    effective_class = get_likely_index(target[:, 0:-2])

    # Check for each element of the batch if the prediction is correct
    equal = predicted_class.eq(effective_class)

    # Counting negative class correct predictions
    negative_class_mask = predicted_class.eq(negative_class_index)
    negative_correct = equal[negative_class_mask].sum().item()

    # Only for correct class predictions of positives, compute IoU
    # select only last two columns which contain bounding box coordinates
    output_bounding_box = output[(equal & ~negative_class_mask), -2:]
    target_bounding_box = target[(equal & ~negative_class_mask), -2:]

    iou = intersection_over_union(output_bounding_box, target_bounding_box)

    return (iou >= iou_threshold).int().sum().item() + negative_correct


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


def get_likely_index(tensor):
    """Find the most likely label index for each element in the batch."""
    return tensor.argmax(dim=-1)


def evaluate(model, criterion, loader, device, iou_threshold, negative_class_index):
    """Evaluate the model on the test/validation set with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []
    pbar = tqdm(total=len(loader), desc="Testing", leave=False)
    # Create a dictionary, to store classification with localization predictions
    # It contains for each class i, (the # of correct predictions for that class + # of total predictions for that class + # of elements in the class)
    n_classes= len(loader.dataset.classes_list)
    dictionary= {i: [] for i in range(n_classes)}


    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            total_loss += criterion(output, target).item()

            # Use number_of_correct to calculate correct predictions
            correct += number_of_correct(output, target, iou_threshold, negative_class_index)

            # Collect predictions and targets for metrics
            predicted_class = get_likely_index(output[:, :-2])
            true_class = get_likely_index(target[:, :-2])
            all_predictions.extend(predicted_class.cpu().numpy())
            all_targets.extend(true_class.cpu().numpy())

            pbar.update(1)

    pbar.close()
    total_loss /= len(loader.dataset)

    # Generate classification report
    report = classification_report(
        all_targets,
        all_predictions,
        target_names=[f"Class {i}" for i in range(len(set(all_targets)))],
        zero_division=0
    )
    print("\nClassification Report:\n", report)

    # Generate classification confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    print("\nConfusion Matrix:\n", conf_matrix)

    # Calculate accuracy
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"\nAccuracy: {accuracy:.2f}%")

    return total_loss, accuracy, report


def initialize_model(architecture,num_classes, device):
    """
    Initialize a ResNet model for classification and bounding box regression.
    """
    if architecture == "ResNet":
        # Load ResNet18
        model = torchvision.models.resnet18(weights=None)
    
        # Modify the first convolutional layer to accept 1 input channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
        # Modify the fully connected layer to output 7 (classes) + 2 (bounding box regression) = 9
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),  # Add a hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes + 2)  # Output: 7 classes + 2 BB regression
        )
        print("Using ResNet18...\n")
    elif architecture == "MobileNet":
        # Load MobileNetV2
        model = torchvision.models.mobilenet_v2(weights=None)
        
        # Modify the first convolutional layer to accept 1 input channel
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        
        # Modify the classifier to output num_classes + 2 (bounding box regression)
        model.classifier = nn.Sequential(
            nn.Linear(model.last_channel, 256),  # Add a hidden layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes + 2)  # Output: num_classes + 2 for bounding box regression
        )
        print("Using MobileNetv2...\n")

    else:
        print("Unsupported architecture. Choose 'ResNet' or 'MobileNet'.")
    # Move to device
    model = model.to(device)
    return model


class CombinedLoss(nn.Module):
    def __init__(self, classes_list, lambda_coord=0.5, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.num_classes = len(classes_list)
        self.lambda_coord = lambda_coord
        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.bb_loss_fn = nn.MSELoss()
        self.classes_list = classes_list

    def forward(self, output, target):
        output_class = output[:, :self.num_classes]
        output_bb = output[:, self.num_classes:]
        target_class = target[:, :self.num_classes]
        target_bb = target[:, self.num_classes:]

        class_loss = self.class_loss_fn(output_class, target_class.argmax(dim=1))

        # We want to ingnore negative class BB
        positive_mask = target_class.argmax(dim=1) != self.classes_list.index('Nonfiller')  
        if positive_mask.any():
            output_bb = output_bb[positive_mask]
            target_bb = target_bb[positive_mask]
            bb_loss = self.bb_loss_fn(output_bb, target_bb)
        else:
            bb_loss = 0.0  # No one positive


        return class_loss + self.lambda_coord * bb_loss
    
class GlobalMSELoss(nn.Module):
    def __init__(self, classes_list, lambda_coord=0.5):
        super(GlobalMSELoss, self).__init__()
        self.num_classes = len(classes_list)
        self.lambda_coord = lambda_coord
        self.class_loss_fn = nn.MSELoss()
        self.bb_loss_fn = nn.MSELoss()
        self.classes_list = classes_list

    def forward(self, output, target):

        output_class = torch.softmax(output[:, :self.num_classes], dim=1)
        #output_class = output[:, :self.num_classes]
        output_bb = output[:, self.num_classes:]
        target_class = target[:, :self.num_classes]
        target_bb = target[:, self.num_classes:]

        #class_loss = self.class_loss_fn(output_class.argmax(dim=1).float(), target_class.argmax(dim=1).float())
        class_loss = self.class_loss_fn(output_class, target_class)


        # We want to ingnore negative class BB
        positive_mask = target_class.argmax(dim=1) != self.classes_list.index('Nonfiller') 
        if positive_mask.any():
            output_bb = output_bb[positive_mask]
            target_bb = target_bb[positive_mask]
            bb_loss = self.bb_loss_fn(output_bb, target_bb)
        else:
            bb_loss = 0.0  # No one positive


        return class_loss + self.lambda_coord * bb_loss


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
    
    # Read the training dataset to get the class order
    train_set = myDataset(root_folder, "train")
    class_order = train_set.classes_list  # Save the class order from the training set
    class_counts = train_set.labels_df.sort_values(by="label")["label"].value_counts(sort=False)
    print(f"Class order (from training set): {class_order}")
    print(f'#occorrenze train_set: {class_counts}\n')

    # Apply the same class order to validation and test datasets
    test_set = myDataset(root_folder, "test")
    validation_set = myDataset(root_folder, "validation")

    batch_size = 64
    train_loader, test_loader, validation_loader = prepare_dataloaders(
        train_set, test_set, validation_set, batch_size, device
    )

    # Model initialization
    num_classes = len(train_set.classes_list)
    architecture = "ResNet"  # Choose between "ResNet" and "MobileNet"
    model = initialize_model(architecture, num_classes, device)
    # Load pre-trained model if available
    best_model_path = "best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}\n")
    else:
        print("No pre-trained model found. Starting training from scratch.\n")
    print(model)
    summary(model, (1, 224, 224))
    start_epoch = 1
    checkpoint_path = "checkpoint.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
        print(f"Loaded checkpoint from {checkpoint_path}\n")
    else:
        print("No checkpoint found. Starting training from scratch.\n")

    # Learning rate
    lr = 0.001
    # Nr. epochs
    n_epochs = 6

    # Loss function, optimizer, and scheduler
    # Calcola i pesi inversamente proporzionali alla frequenza delle classi
    class_counts = torch.tensor(class_counts.values, dtype=torch.float32)
    class_weights = (1.0 / class_counts).to(device)
    #criterion = GlobalMSELoss(classes_list=class_order, lambda_coord=1)
    criterion = CombinedLoss(classes_list=class_order, class_weights=class_weights)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr*10,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
        anneal_strategy="linear",
    )

    # Training and evaluation
    log_interval = train_set.__len__()//(batch_size*100) #stamp every 1%
    losses = []

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )


    best_validation_loss = float("inf")
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(start_epoch, n_epochs + 1):

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
            validation_loss, validation_accuracy, validation_report = evaluate(
                model, criterion, validation_loader, device, iou_threshold=0.5,
                negative_class_index=train_set.classes_dict["Nonfiller"]
            )
            print(f"Validation set: Loss: {validation_loss:.4f}, Accuracy: {validation_accuracy:.2f}%")
            # Saving the best model
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(model.state_dict(), "best_model.pth")
                # Salva il checkpoint
                checkpoint = {
                    'epoch': epoch,  # Salva l'epoca corrente
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                # Esporta il checkpoint in un file
                torch.save(checkpoint, 'checkpoint.pth')
                print(f"Checkpoint salvato all'epoca {epoch} con validation loss {validation_loss:.4f}")

            # Plot the average training loss per epoch
            plt.figure(figsize=(10, 6))
            plt.plot(losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Epochs")
            plt.legend()
            plt.grid(True)
            plt.show(block=False)
            plt.savefig("training_loss.png")
    
    
        pbar.update(1)
    # Final evaluation on the test set
    test_loss, test_accuracy, test_report = evaluate(
        model, criterion, test_loader, device, iou_threshold=0.5,
        negative_class_index=train_set.classes_dict["Nonfiller"]
    )
    print(f"Test set: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")



if __name__ == "__main__":
    main()

