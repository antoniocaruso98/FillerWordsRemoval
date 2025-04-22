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
    model = initialize_model(num_classes, device)

    # Carica i pesi salvati
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print("Best model weights loaded successfully.")

    # Esegui la validazione
    validation_loss, validation_accuracy, validation_report = evaluate(
        model, criterion, validation_loader, device, iou_threshold, negative_class_index
    )


# Esegui la validazione con il modello salvato
def main():
    # Assicurati che il dispositivo sia configurato
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset e DataLoader
    root_folder = os.path.join("..", "Dataset_completo")
    validation_set = myDataset(root_folder, "validation")
    train_set = myDataset(root_folder, "training")
    # Supponendo che tu abbia un'istanza del dataset chiamata train_set

    print(train_set.classes_dict)
    test_set = myDataset(root_folder, "test")
    batch_size = 64
    _, _, validation_loader = prepare_dataloaders(train_set, test_set, validation_set, batch_size, device)

    # Loss function
    num_classes = len(validation_set.classes_list)
    criterion = CombinedLoss(num_classes=num_classes)

    # Indice della classe "Nonfiller"
    negative_class_index = validation_set.classes_dict["Nonfiller"]

    # Percorso del modello salvato
    model_path = "best_model.pth"

    # Esegui la validazione
    validate_with_best_model(model_path, validation_loader, criterion, device, iou_threshold=0.5, negative_class_index=negative_class_index)





if __name__ == "__main__":
    main()