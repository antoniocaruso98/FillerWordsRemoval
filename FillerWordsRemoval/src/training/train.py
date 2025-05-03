import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from models.two_headed_resnet import TwoHeadedResNet
from data.dataset import AudioDataset
from losses.combined_loss import CombinedLoss  # Importa la tua classe CombinedLoss

def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def train(model, criterion, optimizer, train_loader, val_loader, device, num_epochs, save_path):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)

        pbar.close()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validate the model
        val_loss = validate(model, criterion, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    root_folder = os.path.join(os.path.join("..", os.path.join("..", "..")), "DATASET_COMPLETO_V2")
    train_set = AudioDataset(root_folder, "train")
    class_order = train_set.classes_list  # Save the class order from the training set
    class_counts = train_set.labels_df.sort_values(by="label")["label"].value_counts(sort=False)
    print(f"Class order (from training set): {class_order}")
    print(f'#occorrenze train_set: {class_counts}\n')

    val_set = AudioDataset(root_folder, "validation")  # Assuming you have a validation split
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    # Initialize model
    num_classes = len(train_set.classes_list)
    model = TwoHeadedResNet(num_classes=num_classes).to(device)

    # Calculate class weights inversely proportional to occurrences
    class_counts = torch.tensor(class_counts, dtype=torch.float32).to(device)
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()

    # Define combined loss function
    criterion = CombinedLoss(classes_list=class_order, lambda_coord=0.5, class_weights=class_weights)

    # Load pre-trained weights
    #pretrained_weights_path = os.path.join("..","pretrained_weights", "audio_dataset_weights.pth")
    #model.load_pretrained_weights(pretrained_weights_path=None)  # Replace with actual path if needed

    # Freeze layers for fine-tuning
    #for param in model.parameters():
        #param.requires_grad = False
    #for param in model.fc.parameters():  # Unfreeze only the final layer
        #param.requires_grad = True

    # Define optimizer
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Train the model
    num_epochs = 10
    save_path = "best_model.pth"  # Path to save the best model
    train(model, criterion, optimizer, train_loader, val_loader, device, num_epochs, save_path)

if __name__ == "__main__":
    main()