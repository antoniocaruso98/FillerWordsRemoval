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


def train(model, criterion, optimizer, epoch, log_interval, train_loader, device, scheduler):
    """Train the model for one epoch."""
    model.train()
    losses = []
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss,class_loss,delta_loss, center_loss, coherence_loss = criterion(output, target)

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
                f"- Total Loss: {loss.item():.4f} | Class Loss: {class_loss.item():.4f} | Delta Loss: {delta_loss.item():.4f} | Center Loss: {center_loss.item():.4f} | Coherence Loss: {coherence_loss.item():.4f}"
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

    return (iou >= iou_threshold).int().sum().item() + negative_correct


def intersection_over_union(output_bb, target_bb):
    """
    Computes IoU between output and target bounding boxes for each sample.
    Bounding boxes are given in (center, width) format.
    """
    # Convert (center, width) to (start, end)
    output_se = torch.zeros_like(output_bb)
    target_se = torch.zeros_like(target_bb)
    
    output_se[:, 0] = output_bb[:, 1] - output_bb[:, 0] / 2
    output_se[:, 1] = output_bb[:, 1] + output_bb[:, 0] / 2

    target_se[:, 0] = target_bb[:, 1] - target_bb[:, 0] / 2
    target_se[:, 1] = target_bb[:, 1] + target_bb[:, 0] / 2

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
    all_targets = []
    all_predictions = []
    all_durations = []
    pred_durations = []
    center_errors = []
    delta_errors = []
    relative_errors = []
    # Per MAE e MSE
    all_output_bb = []
    all_target_bb = []
    pbar = tqdm(total=len(loader), desc="Evaluating", leave=False)

    iou_list = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss,_,_,_,_ = criterion(output, target)
            total_loss += loss.item()

            # Use number_of_correct to calculate correct predictions
            correct += number_of_correct(output, target, iou_threshold, negative_class_index)

            # Collect predictions and targets for metrics
            predicted_class = get_likely_index(output[:, :-2])
            true_class = get_likely_index(target[:, :-2])
            positive_mask = (true_class != negative_class_index)
            # Only for positive classes calculate regression metrics
            if positive_mask.any():
                output_bb = output[positive_mask, -2:]
                target_bb = target[positive_mask, -2:]

                # Accumulate bounding box predictions and targets
                all_output_bb.append(output_bb.cpu())
                all_target_bb.append(target_bb.cpu())

                # Transform center and width to xmin, xmax
                output_xmin = output_bb[:, 1] - (output_bb[:, 0] / 2)
                output_xmax = output_bb[:, 1] + (output_bb[:, 0] / 2)
                target_xmin = target_bb[:, 1] - (target_bb[:, 0] / 2)
                target_xmax = target_bb[:, 1] + (target_bb[:, 0] / 2)

                # Calculate durations
                pred_duration = output_xmax - output_xmin
                true_duration = target_xmax - target_xmin
                # Collect durations for metrics
                pred_durations.extend(pred_duration.cpu().numpy())
                all_durations.extend(true_duration.cpu().numpy())
                # Calculate relative error
                relative_error = torch.abs(pred_duration - true_duration) / true_duration
                relative_errors.extend(relative_error.cpu().numpy())

                # Calculate delta error
                delta_error = torch.abs(output_bb[:, 0] - target_bb[:, 0])
                delta_errors.extend(delta_error.cpu().numpy())
                # Calculate center error
                center_error = torch.abs(output_bb[:, 1] - target_bb[:, 1])
                center_errors.extend(center_error.cpu().numpy())

                # Calculate IoU for positives
                iou_values = intersection_over_union(output_bb, target_bb)
                iou_list.extend(iou_values.cpu().numpy().tolist())

            all_predictions.extend(predicted_class.cpu().numpy())
            all_targets.extend(true_class.cpu().numpy())

            pbar.update(1)

    pbar.close()
    total_loss /= len(loader)

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


    # MAE and MSE for center and delta
    all_output_bb = torch.cat(all_output_bb, dim=0)
    all_target_bb = torch.cat(all_target_bb, dim=0)
    mae_center = mean_absolute_error(all_target_bb[:, 1].numpy(), all_output_bb[:, 1].numpy())
    mse_center = mean_squared_error(all_target_bb[:, 1].numpy(), all_output_bb[:, 1].numpy())
    mae_delta = mean_absolute_error(all_target_bb[:, 0].numpy(), all_output_bb[:, 0].numpy())
    mse_delta = mean_squared_error(all_target_bb[:, 0].numpy(), all_output_bb[:, 0].numpy())
    print(f"\nMAE Center: {mae_center:.4f}, MSE Center: {mse_center:.4f}")
    print(f"MAE Delta: {mae_delta:.4f}, MSE Delta: {mse_delta:.4f}")
    normalized_mae_center = mae_center / np.mean(all_target_bb[:, 1].numpy())
    normalized_mae_delta = mae_delta / np.mean(all_target_bb[:, 0].numpy())
    print(f"Normalized MAE Center: {normalized_mae_center:.4f}")
    print(f"Normalized MAE Delta: {normalized_mae_delta:.4f}")

    # Calculate overall Accuracy
    accuracy = 100.0 * correct / len(loader.dataset)
    print(f"Accuracy (Classification+IoU>0.5): {accuracy:.2f}%")

    # Calculate mean IoU
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    print(f"Mean IoU (positives): {mean_iou:.4f}")

    # Calculate relative error metrics
    if relative_errors:
        mean_relative_error = np.mean(relative_errors)
        print(f"Mean Relative Error (Duration): {mean_relative_error:.4f}")
        mean_target_duration = np.mean(all_durations) # mean of real target durations
        error_in_seconds = mean_relative_error * mean_target_duration
        print(f"Errore medio sulla durata: {error_in_seconds:.4f} secondi")

    # Calculate duration accuracy
    if all_durations and pred_durations:
        duration_accuracy = np.mean(
            np.abs(np.array(pred_durations) - np.array(all_durations)) / np.array(all_durations) < 0.1
        )
        print(f"Duration Accuracy (within 10%): {duration_accuracy:.2f}")
    
    # Analyze error distributions for center
    print("\nError Analysis (Center):")
    print(f"Mean Offset Error (Center): {np.mean(center_errors):.4f}")
    print(f"Max Offset Error (Center): {np.max(center_errors):.4f}")
    print(f"Min Offset Error (Center): {np.min(center_errors):.4f}")

    # Analyze error distributions for delta
    print("\nError Analysis (Delta):")
    print(f"Mean Error (Delta): {np.mean(delta_errors):.4f}")
    print(f"Max Error (Delta): {np.max(delta_errors):.4f}")
    print(f"Min Error (Delta): {np.min(delta_errors):.4f}")

    # Plot error distributions
    plt.figure(figsize=(10, 6))
    plt.hist(center_errors, bins=50, alpha=0.7, label="Center Offset Errors")
    plt.hist(delta_errors, bins=50, alpha=0.7, label="Delta Errors")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Center and Delta Errors")
    plt.legend()
    plt.grid(True)
    plt.savefig("center_delta_error_distribution.png")
    print("Saved center and delta error distribution to 'center_delta_error_distribution.png'")

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

        # Add two separate heads for classification and regression
        class_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Output: num_classes for classification
        )
        bb_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # Output: 2 (delta, center)
        )

        print("Using ResNet18 with two heads...\n")

    elif architecture == "MobileNet":
        # Load MobileNetV2
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 1 input channel
        model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1, bias=False)

        # Freeze all layers in the feature extractor
        for param in model.features.parameters():
            param.requires_grad = False

        # Remove the original classifier
        num_features = model.last_channel
        model.classifier = nn.Identity()  # Replace with identity to extract features

        # Add two separate heads for classification and regression
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

        print("Using MobileNetV2 with frozen feature extractor and two heads...\n")

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
    def __init__(self, classes_list, lambda_center, lambda_delta, lambda_coherence, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.num_classes = len(classes_list)
        self.lambda_center = lambda_center
        self.lambda_delta = lambda_delta
        self.lambda_coherence = lambda_coherence
        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.delta_loss_fn = nn.SmoothL1Loss()  # Loss for delta
        self.center_loss_fn = nn.SmoothL1Loss()  # Loss for center
        self.classes_list = classes_list

    def forward(self, output, target):
        output_class = output[:, :self.num_classes]
        output_delta = output[:, self.num_classes]  # Larghezza (delta)
        output_center = output[:, self.num_classes + 1]  # Centro
        target_class = target[:, :self.num_classes]
        target_delta = target[:, self.num_classes]  # Larghezza (delta)
        target_center = target[:, self.num_classes + 1]  # Centro


        class_loss = self.class_loss_fn(output_class, target_class.argmax(dim=1))

        # We want to ingnore negative class BB
        positive_mask = target_class.argmax(dim=1) != self.classes_list.index('Nonfiller')  
        if positive_mask.any():
            output_delta = output_delta[positive_mask]
            output_center = output_center[positive_mask]
            target_delta = target_delta[positive_mask]
            target_center = target_center[positive_mask]

            delta_loss = self.delta_loss_fn(output_delta, target_delta)
            center_loss = self.center_loss_fn(output_center, target_center)

            # Coherence loss
            output_xmin = output_center - (output_delta / 2)
            output_xmax = output_center + (output_delta / 2)
            target_xmin = target_center - (target_delta / 2)
            target_xmax = target_center + (target_delta / 2)
            coherence_loss = nn.SmoothL1Loss()(output_xmin, target_xmin) + nn.SmoothL1Loss()(output_xmax, target_xmax)

        else:
            delta_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
            center_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
            coherence_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)


        # Compute the total loss
        total_loss= class_loss + self.lambda_center * center_loss + self.lambda_delta * delta_loss + self.lambda_coherence * coherence_loss
        return total_loss, class_loss, self.lambda_delta * delta_loss, self.lambda_center *center_loss, self.lambda_coherence * coherence_loss
    

class DynamicCombinedLoss(nn.Module):
    def __init__(self, classes_list, init_lambda_center, init_lambda_delta, init_lambda_coherence, class_weights=None):
        """
        init_lambda_* sono i valori iniziali che si utilizzeranno per inizializzare i parametri dinamici.
        Questi verranno trasformati in log sigma per stabilità numerica e per permettere la loro ottimizzazione.
        """
        super(DynamicCombinedLoss, self).__init__()
        self.num_classes = len(classes_list)
        self.classes_list = classes_list

        # Loss per la classificazione (si mantiene invariata)
        self.class_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        
        # Loss per le regressioni (delta e center)
        self.delta_loss_fn = nn.SmoothL1Loss()
        self.center_loss_fn = nn.SmoothL1Loss()
        
        # Parametri dinamici per pesare le componenti di regressione:
        # Inizializziamo i parametri come log(sigma), in modo da garantire sigma > 0
        self.log_sigma_center = nn.Parameter(torch.log(torch.tensor(init_lambda_center, dtype=torch.float)))
        self.log_sigma_delta = nn.Parameter(torch.log(torch.tensor(init_lambda_delta, dtype=torch.float)))
        self.log_sigma_coherence = nn.Parameter(torch.log(torch.tensor(init_lambda_coherence, dtype=torch.float)))

    def forward(self, output, target):
        # Estrazione delle componenti: la prima parte per la classificazione,
        # le successive per le componenti di regressione
        output_class = output[:, :self.num_classes]
        output_delta = output[:, self.num_classes]       # Larghezza (delta)
        output_center = output[:, self.num_classes + 1]    # Centro
        
        target_class = target[:, :self.num_classes]
        target_delta = target[:, self.num_classes]         # Larghezza (delta)
        target_center = target[:, self.num_classes + 1]      # Centro

        # Loss per la classificazione
        class_loss = self.class_loss_fn(output_class, target_class.argmax(dim=1))

        # Creiamo una maschera per escludere i bounding box non positivi sulla base della classe 'Nonfiller'
        positive_mask = target_class.argmax(dim=1) != self.classes_list.index('Nonfiller')
        
        if positive_mask.any():
            output_delta_pos = output_delta[positive_mask]
            output_center_pos = output_center[positive_mask]
            target_delta_pos = target_delta[positive_mask]
            target_center_pos = target_center[positive_mask]

            # Calcolo delle loss per delta e center
            delta_loss = self.delta_loss_fn(output_delta_pos, target_delta_pos)
            center_loss = self.center_loss_fn(output_center_pos, target_center_pos)

            # Coherence loss: calcoliamo xmin e xmax dai centri e dagli delta
            output_xmin = output_center_pos - (output_delta_pos / 2)
            output_xmax = output_center_pos + (output_delta_pos / 2)
            target_xmin = target_center_pos - (target_delta_pos / 2)
            target_xmax = target_center_pos + (target_delta_pos / 2)
            
            # Usiamo lo stesso tipo di loss (SmoothL1) per la componente di coerenza
            coherence_loss = self.delta_loss_fn(output_xmin, target_xmin) + self.delta_loss_fn(output_xmax, target_xmax)
        else:
            delta_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
            center_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)
            coherence_loss = torch.tensor(0.0, device=output.device, dtype=output.dtype)

        # Calcoliamo sigma da log_sigma (per garantire positività)
        sigma_center = torch.exp(self.log_sigma_center)
        sigma_delta = torch.exp(self.log_sigma_delta)
        sigma_coherence = torch.exp(self.log_sigma_coherence)

        # Pesi dinamici per ogni loss: formulazione uncertainty weighting
        weighted_center_loss = center_loss / (2 * sigma_center ** 2) + self.log_sigma_center
        weighted_delta_loss = delta_loss / (2 * sigma_delta ** 2) + self.log_sigma_delta
        weighted_coherence_loss = coherence_loss / (2 * sigma_coherence ** 2) + self.log_sigma_coherence

        # La loss totale è la somma della loss per la classificazione e quelle pesate per le regressioni
        total_loss = class_loss + weighted_center_loss + weighted_delta_loss + weighted_coherence_loss

        return total_loss, class_loss, weighted_delta_loss, weighted_center_loss, weighted_coherence_loss



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

    # LOSS function
    # Calcola i pesi inversamente proporzionali alla frequenza delle classi
    class_counts = torch.tensor(class_counts.values, dtype=torch.float32)
    class_weights = (1.0 / class_counts).to(device)
    #criterion = GlobalMSELoss(classes_list=class_order, lambda_coord=1)
    lambda_coord = 25*2
    #criterion = CombinedLoss(classes_list=class_order, class_weights=class_weights, lambda_center=lambda_coord, lambda_delta=2*lambda_coord, lambda_coherence=lambda_coord)
    criterion = DynamicCombinedLoss(classes_list=class_order, init_lambda_center=lambda_coord, init_lambda_delta=2*lambda_coord, init_lambda_coherence=lambda_coord, class_weights=class_weights).to(device)

    # Training parameters
    #lr = 0.00001 #low
    lr= 0.0001 #optimal
    total_epochs = 50
    # Scheduler and Optimizer
    #optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=lr, weight_decay=0.0001)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr * 10,
        steps_per_epoch=len(train_loader),
        epochs=total_epochs,
        anneal_strategy="linear",
    )
    # Checkpoint 
    checkpoint_path = os.path.join("..", "checkpoint.pth")
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint caricato: epoca {checkpoint['epoch']}\n")
    else:
        print("Nessun checkpoint trovato. Inizio del training da zero.\n")
        


    
    # Training and evaluation
    log_interval = len(train_loader)//100 #stamp every 1%
    losses = []
    validation_losses = []

    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    best_validation_loss = float("inf")
    plot_file = "training_loss.png"
    with tqdm(total=total_epochs, initial=start_epoch-1) as pbar:
        for epoch in range(start_epoch, total_epochs + 1):

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
            validation_losses.append(validation_loss)
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
            plt.plot(range(start_epoch, start_epoch + len(losses)), losses, label="Training Loss", color="blue")
            plt.plot(range(start_epoch, start_epoch + len(losses)), validation_losses, label="Validation Loss", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Over Epochs")
            plt.legend()
            plt.grid(True)
            # Salva il grafico aggiornato
            plt.savefig(plot_file)
            print(f"Grafico aggiornato salvato in {plot_file}")

            # Monitor the learning rate
            current_lr = scheduler.get_last_lr()[0]
            print(f"Current Learning Rate: {current_lr:.6f}")
    
            pbar.update(1)

    # Final evaluation on the test set
    test_loss = evaluate(
        model, criterion, test_loader, device, iou_threshold=0.5,
        negative_class_index=train_set.classes_dict["Nonfiller"]
    )
    print(f"Test set: Loss: {test_loss:.4f}")



if __name__ == "__main__":
    main()

