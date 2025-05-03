# filepath: FillerWordsRemoval/src/utils/helpers.py
import torch
import numpy as np
import matplotlib.pyplot as plt

def normalize_tensor(tensor):
    """Normalize a tensor to have zero mean and unit variance."""
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

def calculate_accuracy(predictions, targets):
    """Calculate the accuracy of predictions against targets."""
    correct = (predictions.argmax(dim=1) == targets).float().sum()
    return correct / targets.size(0)

def plot_loss(losses, title='Loss over Epochs'):
    """Plot the training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def save_model(model, path):
    """Save the model state to a specified path."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load the model state from a specified path."""
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model

def compute_iou(box_a, box_b):
    """Compute the Intersection over Union (IoU) for two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0