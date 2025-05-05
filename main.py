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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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

        # Apply data augmentation only for training set
        if self.type == "train":
            if np.random.rand() > 0.5:  # Randomly scale volume
                gain = np.random.uniform(0.9, 1.1)  # Scale volume by 0.9x to 1.1x
                audio = audio * gain
                
            if np.random.rand() > 0.5:  # Randomly apply pitch shifting
                n_steps = np.random.uniform(-1, 1)  # Shift pitch by -2 to +2 semitones
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

            if np.random.rand() > 0.5:  # Randomly apply time stretching
                rate = np.random.uniform(0.9, 1.1)  # Stretch by 0.8x to 1.2x
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


def train(model, criterion, optimizer, epoch, log_interval, train_loader, device, scheduler):
    """Train the model for one epoch."""
    model.train()
    losses = []
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss,class_loss,bb_loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the learning rate scheduler
        scheduler.step()

        # Log training stats
        if batch_idx % log_interval == 0:
            tqdm.write(
                f"[Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} "
                f"- Total Loss: {loss.item():.4f} | Class Loss: {class_loss.item():.4f} | BB Loss: {bb_loss.item():.4f}"
            )

        # Update progress bar and record loss
        pbar.update(1)
        losses.append(loss.item())

    pbar.close()
    print(f"[Epoch {epoch}] Average Training Loss: {np.mean(losses):.4f}")
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

    return (iou >= iou_threshold).int().sum().item() #+ negative_correct


def intersection_over_union(output_bb, target_bb):
    """
    Computes IoU between output and target bounding boxes for each sample.
    Bounding boxes are given in (center, width) format.
    """
    # Convert (center, width) to (start, end)
    output_se = torch.zeros_like(output_bb)
    target_se = torch.zeros_like(target_bb)
    
    output_se[:, 0] = output_bb[:, 0] - output_bb[:, 1] / 2
    output_se[:, 1] = output_bb[:, 0] + output_bb[:, 1] / 2

    target_se[:, 0] = target_bb[:, 0] - target_bb[:, 1] / 2
    target_se[:, 1] = target_bb[:, 0] + target_bb[:, 1] / 2

    # Compute intersection
    inter_start = torch.max(output_se[:, 0], target_se[:, 0])
    inter_end = torch.min(output_se[:, 1], target_se[:, 1])
    intersection = torch.clamp(inter_end - inter_start, min=0)

    # Compute union: width_A + width_B - intersection
    width_output = output_se[:, 1] - output_se[:, 0]
    width_target = target_se[:, 1] - target_se[:, 0]
    union = width_output + width_target - intersection

    return intersection / union


def get_likely_index(tensor):
    """Find the most likely label index for each element in the batch."""
    return tensor.argmax(dim=-1)


def evaluate(model, criterion, loader, device, iou_threshold, negative_class_index):
    """Evaluate the model on the test/validation set with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    positive_total = 0
    all_targets = []
    all_predictions = []
    all_durations = []
    pred_durations = []
    offset_errors = []
    relative_errors = []
    pbar = tqdm(total=len(loader), desc="Evaluating", leave=False)

    iou_list = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss,class_loss,bb_loss = criterion(output, target)
            total_loss += loss.item()

            # Use number_of_correct to calculate correct predictions
            correct += number_of_correct(output, target, iou_threshold, negative_class_index)

            # Collect predictions and targets for metrics
            predicted_class = get_likely_index(output[:, :-2])
            true_class = get_likely_index(target[:, :-2])
            positive_mask = (true_class != negative_class_index)

            if positive_mask.any():
                output_bb = output[positive_mask, -2:]
                target_bb = target[positive_mask, -2:]

                # Transform center and width to xmin, xmax
                output_xmin = output_bb[:, 0] - (output_bb[:, 1] / 2)
                output_xmax = output_bb[:, 0] + (output_bb[:, 1] / 2)
                target_xmin = target_bb[:, 0] - (target_bb[:, 1] / 2)
                target_xmax = target_bb[:, 0] + (target_bb[:, 1] / 2)

                # Calculate durations
                pred_duration = output_xmax - output_xmin
                true_duration = target_xmax - target_xmin

                # Collect durations for metrics
                pred_durations.extend(pred_duration.cpu().numpy())
                all_durations.extend(true_duration.cpu().numpy())

                # Calculate relative error
                relative_error = torch.abs(pred_duration - true_duration) / true_duration
                relative_errors.extend(relative_error.cpu().numpy())

                # Calculate offset error
                offset_error = torch.abs(output_bb[:, 0] - target_bb[:, 0])
                offset_errors.extend(offset_error.cpu().numpy())

                # Calculate IoU for positive samples
                iou_values = intersection_over_union(output_bb, target_bb)
                iou_list.extend(iou_values.cpu().numpy().tolist())

            all_predictions.extend(predicted_class.cpu().numpy())
            all_targets.extend(true_class.cpu().numpy())

            all_positive_mask = (true_class != negative_class_index)
            positive_total += all_positive_mask.sum().item()

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

    # Calculate IoU accuracy
    accuracy = 100.0 * correct / positive_total
    print(f"\nIoU Accuracy (positive classes): {accuracy:.2f}%")

    # Calculate mean IoU
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    print(f"Mean IoU (positives): {mean_iou:.4f}")

    # Calculate relative error metrics
    if relative_errors:
        mean_relative_error = np.mean(relative_errors)
        print(f"Mean Relative Error (Duration): {mean_relative_error:.4f}")

    # Calculate offset error metrics
    if offset_errors:
        mean_offset_error = np.mean(offset_errors)
        print(f"Mean Offset Error (Center): {mean_offset_error:.4f}")

    # Calculate duration accuracy
    if all_durations and pred_durations:
        duration_accuracy = np.mean(
            np.abs(np.array(pred_durations) - np.array(all_durations)) / np.array(all_durations) < 0.1
        )
        print(f"Duration Accuracy (within 10%): {duration_accuracy:.2f}")

    return total_loss


def initialize_model(architecture, num_classes, device):
    """
    Initialize a model with two heads: one for classification and one for bounding box regression.
    """
    if architecture == "ResNet":
        # Load ResNet18
        model = torchvision.models.resnet18(weights=None)

        # Modify the first convolutional layer to accept 1 input channel
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Remove the original fully connected layer
        num_features = model.fc.in_features
        model.fc = nn.Identity()  # Replace with identity to extract features

        # Add two separate heads
        class_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Output: num_classes for classification
        )
        bb_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # Output: 2 for bounding box regression
        )

        print("Using ResNet18 with two heads...\n")

    elif architecture == "MobileNet":
        # Load MobileNetV2
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 1 input channel
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=False)

        # Remove the original classifier
        num_features = model.last_channel
        model.classifier = nn.Identity()  # Replace with identity to extract features

        # Add two separate heads
        class_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Output: num_classes for classification
        )
        bb_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # Output: 2 for bounding box regression
        )

        print("Using MobileNetV2 with two heads...\n")

    else:
        raise ValueError("Unsupported architecture. Choose 'ResNet' or 'MobileNet'.")

    # Define a combined model with two heads
    class TwoHeadedModel(nn.Module):
        def __init__(self, base_model, class_head, bb_head):
            super(TwoHeadedModel, self).__init__()
            self.base_model = base_model
            self.class_head = class_head
            self.bb_head = bb_head

        def forward(self, x):
            features = self.base_model(x)  # Extract shared features
            class_output = self.class_head(features)  # Classification head
            bb_output = self.bb_head(features)  # Bounding box regression head
            return torch.cat((class_output, bb_output), dim=1)  # Concatenate outputs

    # Instantiate the combined model and move it to the device
    combined_model = TwoHeadedModel(model, class_head, bb_head).to(device)
    return combined_model

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
            bb_loss = torch.tensor(0.0, device=output_bb.device, dtype=output_bb.dtype) # No one positive tensor


        total_loss= class_loss + self.lambda_coord * bb_loss
        return total_loss,class_loss,self.lambda_coord * bb_loss
    
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
        num_workers=num_workers/2,
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



def main():

    # Starting with always the same seed for weights reproducibility
    torch.manual_seed(42)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Dataset
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
    # Dataloaders
    batch_size = 64
    num_workers = 8 # CPU architecture
    train_loader, test_loader, validation_loader = prepare_dataloaders(
        train_set, test_set, validation_set, batch_size, device, num_workers=num_workers
    )

    # Model initialization
    num_classes = len(train_set.classes_list)
    architecture = "ResNet"  # Choose between "ResNet" and "MobileNet"
    model = initialize_model(architecture, num_classes, device)
    print(model)
    summary(model, (1, 224, 224))

    lr = 0.001
    # Nr.epochs to do
    n_epochs = 10

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    start_epoch = 1
    checkpoint_path = os.path.join("..","checkpoint.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # Riprendi dalla prossima epoca
        print(f"Loaded checkpoint from {checkpoint_path}\n")
        # Calcola il totale delle epoche: epoche già fatte + epoche aggiuntive
        total_epochs = (start_epoch - 1) + n_epochs
        # Imposta last_epoch sottraendo 1 per compensare il passo iniziale automatico
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            steps_per_epoch=len(train_loader),
            epochs=total_epochs,
            anneal_strategy="linear",
            last_epoch=(start_epoch - 1) * len(train_loader) - 1,
            pct_start=0.1,  # Riduce la fase di warm-up per non "sprecare" le prime epoche
        )
        # Se vuoi, non caricare lo state_dict dello scheduler (dato che i parametri totali sono cambiati)
    else:
        print("No checkpoint found. Starting training from scratch.\n")
        total_epochs = n_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            steps_per_epoch=len(train_loader),
            epochs=total_epochs,
            anneal_strategy="linear",
        )

    # Loss function
    # Calcola i pesi inversamente proporzionali alla frequenza delle classi
    class_counts = torch.tensor(class_counts.values, dtype=torch.float32)
    class_weights = (1.0 / class_counts).to(device)
    #criterion = GlobalMSELoss(classes_list=class_order, lambda_coord=1)
    lambda_coord = 25
    criterion = CombinedLoss(classes_list=class_order, class_weights=class_weights, lambda_coord=lambda_coord)

    # Training and evaluation
    log_interval = len(train_loader)//100 #stamp every 1%
    losses = []

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    best_validation_loss = float("inf")
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(start_epoch, start_epoch+n_epochs):

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
            validation_loss = evaluate(
                model, criterion, validation_loader, device, iou_threshold=0.5,
                negative_class_index=train_set.classes_dict["Nonfiller"]
            )
            print(f"Validation set: Loss: {validation_loss:.4f}")
            # Saving the best model
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                # Salva il checkpoint
                checkpoint = {
                    'epoch': epoch,  # Salva l'epoca corrente
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                # Esporta il checkpoint in un file
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint salvato all'epoca {epoch} con validation loss {validation_loss:.4f}")

            # Plot the average training loss per epoch
            plt.figure(figsize=(10, 6))
            plt.plot(losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Epochs")
            plt.legend()
            plt.grid(True)
            #plt.show(block=False)
            plt.savefig("training_loss.png")
    
    
        pbar.update(1)
    # Final evaluation on the test set
    test_loss = evaluate(
        model, criterion, test_loader, device, iou_threshold=0.5,
        negative_class_index=train_set.classes_dict["Nonfiller"]
    )
    print(f"Test set: Loss: {test_loss:.4f}")



if __name__ == "__main__":
    main()

