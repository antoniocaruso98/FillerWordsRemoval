import torch
import torch.nn as nn

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

        # Classification loss
        class_loss = self.class_loss_fn(output_class, target_class.argmax(dim=1))

        # Bounding box loss (only for positive samples)
        positive_mask = target_class.argmax(dim=1) != self.classes_list.index('Nonfiller')
        if positive_mask.any():
            output_bb = output_bb[positive_mask]
            target_bb = target_bb[positive_mask]
            bb_loss = self.bb_loss_fn(output_bb, target_bb)
        else:
            bb_loss = 0.0

        return class_loss + self.lambda_coord * bb_loss