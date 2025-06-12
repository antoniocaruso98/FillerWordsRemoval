import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from main import myDataset, initialize_model, intersection_over_union

def evaluate_from_checkpoint(checkpoint_path, dataset_path, batch_size=64, device="cuda"):
    # Imposta il dispositivo
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Carica il dataset di test
    test_set = myDataset(dataset_path, "validation")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Inizializza il modello
    num_classes = len(test_set.classes_list)
    model = initialize_model("ResNet", num_classes, device)

    # Carica il checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint non trovato: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Checkpoint caricato: epoca {checkpoint['epoch']}")

    # Valutazione
    all_predictions = []
    all_targets = []
    all_output_bb = []
    all_target_bb = []
    iou_list = []
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Estrai le predizioni
            predicted_class = output[:, :-2].argmax(dim=1)
            true_class = target[:, :-2].argmax(dim=1)

            # Confronta le bounding box
            output_bb = output[:, -2:]  # Predette (center, delta)
            target_bb = target[:, -2:]  # Ground truth (center, delta)
            iou = intersection_over_union(output_bb, target_bb)
            iou_list.extend(iou.cpu().numpy())

            # Salva predizioni, ground truth e bounding box
            all_predictions.extend(predicted_class.cpu().numpy())
            all_targets.extend(true_class.cpu().numpy())
            all_output_bb.extend(output_bb.cpu().numpy())
            all_target_bb.extend(target_bb.cpu().numpy())

    # Calcola metriche
    iou_mean = np.mean(iou_list)
    print(f"Mean IoU: {iou_mean:.4f}")

    # Confronta predizioni e ground truth
    print("\nConfronto tra predizioni, bounding box e ground truth:")
    for pred, gt, pred_bb, gt_bb in zip(all_predictions[:10], all_targets[:10], all_output_bb[:10], all_target_bb[:10]):  # Mostra solo i primi 10 esempi
        print(f"Predizione: {pred}, Ground Truth: {gt}")
        print(f"Bounding Box Predetta (center, delta): {pred_bb}")
        print(f"Bounding Box Ground Truth (center, delta): {gt_bb}")
        print("-" * 50)

if __name__ == "__main__":
    # Percorso al checkpoint e al dataset
    checkpoint_path = os.path.join("..", "checkpoint.pth")
    dataset_path = os.path.join("..", "DATASET_COMPLETO_V2")

    # Esegui la valutazione
    evaluate_from_checkpoint(checkpoint_path, dataset_path)