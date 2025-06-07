import os
import torch
from main import myDataset  # Assicurati che la classe myDataset sia definita correttamente
from torch.utils.data import DataLoader
from main import CombinedLoss  # Assicurati che CombinedLoss sia definita correttamente
from main import initialize_model  # Assicurati che initialize_model sia definita correttamente
from main import evaluate  # Assicurati che evaluate sia definita correttamente
from main import prepare_dataloaders  # Assicurati che prepare_dataloaders sia definita correttamente


def validate_with_best_model(model_path, validation_loader, criterion, device, iou_threshold, negative_class_index):
    """
    Load the best model weights and perform validation.
    """
    # Inizializza il modello
    num_classes = len(validation_loader.dataset.classes_list)
    model = initialize_model("ResNet34",num_classes, device)

    # Carica i pesi salvati
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    print("Best model weights loaded successfully.")

    # Esegui la validazione
    validation_loss = evaluate(
        model, criterion, validation_loader, device, iou_threshold, negative_class_index
    )


# Esegui la validazione con il modello salvato
def main():
    # Assicurati che il dispositivo sia configurato
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset e DataLoader
    root_folder = os.path.join("..", "DATASET_COMPLETO_V3")
    # Read the training dataset to get the class order
    train_set = myDataset(root_folder, "train")
    class_order = train_set.classes_list  # Save the class order from the training set
    print(f"Class order (from training set): {class_order}")
    # Apply the same class order to validation and test datasets
    test_set = myDataset(root_folder, "test")
    validation_set = myDataset(root_folder, "validation")


    batch_size = 64
    _, test_loader, validation_loader = prepare_dataloaders(train_set, test_set, validation_set, batch_size, device)
 
   
    # Loss function
    class_counts = train_set.labels_df.sort_values(by="label")["label"].value_counts(sort=False)
    class_counts = torch.tensor(class_counts.values, dtype=torch.float32)
    class_weights = (1.0 / class_counts).to(device)
    #criterion = GlobalMSELoss(classes_list=class_order, lambda_coord=1)
    lambda_coord = 25*2
    criterion = CombinedLoss(classes_list=class_order, class_weights=class_weights, lambda_center=lambda_coord, lambda_delta=2*lambda_coord, lambda_coherence=0.5*lambda_coord)
    

    # Indice della classe "Nonfiller"
    negative_class_index = validation_set.classes_dict["Nonfiller"]

    # Percorso del modello salvato
    model_path = os.path.join("..","checkpoint.pth")
    #model_path = os.path.join("results", "ResNet4.pth")

    # Esegui la validazione
    validate_with_best_model(model_path, test_loader, criterion, device, iou_threshold=0.5, negative_class_index=negative_class_index)





if __name__ == "__main__":
    main()