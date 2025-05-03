import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, criterion, loader, device, iou_threshold, negative_class_index):
    model.eval()
    total_loss = 0
    correct = 0
    positive_total = 0
    all_targets = []
    all_predictions = []
    
    iou_list = []
    bb_absolute_error = []

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
            positive_mask = (true_class != negative_class_index)
            
            if positive_mask.any():
                output_bb = output[positive_mask, -2:]
                target_bb = target[positive_mask, -2:]
                iou_values = intersection_over_union(output_bb, target_bb)
                iou_list.extend(iou_values.cpu().numpy().tolist())
                abs_error = torch.abs(output_bb - target_bb)
                bb_absolute_error.extend(abs_error.mean(dim=1).cpu().numpy().tolist())

            all_predictions.extend(predicted_class.cpu().numpy())
            all_targets.extend(true_class.cpu().numpy())

            all_positive_mask = (true_class != negative_class_index)
            positive_total += all_positive_mask.sum().item()

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

    # Calculate mean IoU and mean bounding box absolute error if there are positive samples
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    mean_bb_error = np.mean(bb_absolute_error) if bb_absolute_error else 0.0
    print(f"Mean IoU (positives): {mean_iou:.4f}")
    print(f"Mean Bounding Box Absolute Error: {mean_bb_error:.4f}")

    return total_loss, accuracy, report

def number_of_correct(output, target, iou_threshold, negative_class_index):
    predicted_class = get_likely_index(output[:, :-2])
    effective_class = get_likely_index(target[:, :-2])
    equal = predicted_class.eq(effective_class)

    negative_class_mask = predicted_class.eq(negative_class_index)
    negative_correct = equal[negative_class_mask].sum().item()

    output_bounding_box = output[(equal & ~negative_class_mask), -2:]
    target_bounding_box = target[(equal & ~negative_class_mask), -2:]

    iou = intersection_over_union(output_bounding_box, target_bounding_box)

    return (iou >= iou_threshold).int().sum().item() + negative_correct

def intersection_over_union(output_bb, target_bb):
    output_se = torch.zeros_like(output_bb)
    target_se = torch.zeros_like(target_bb)
    
    output_se[:, 0] = output_bb[:, 0] - output_bb[:, 1] / 2
    output_se[:, 1] = output_bb[:, 0] + output_bb[:, 1] / 2

    target_se[:, 0] = target_bb[:, 0] - target_bb[:, 1] / 2
    target_se[:, 1] = target_bb[:, 0] + target_bb[:, 1] / 2

    inter_start = torch.max(output_se[:, 0], target_se[:, 0])
    inter_end = torch.min(output_se[:, 1], target_se[:, 1])
    intersection = torch.clamp(inter_end - inter_start, min=0)

    width_output = output_se[:, 1] - output_se[:, 0]
    width_target = target_se[:, 1] - target_se[:, 0]
    union = width_output + width_target - intersection

    return intersection / union

def get_likely_index(tensor):
    return tensor.argmax(dim=-1)