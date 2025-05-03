import torch
import torch.nn as nn
import torchvision.models as models

class TwoHeadedResNet(nn.Module):
    def __init__(self, num_classes, pretrained_weights_path=None):
        super(TwoHeadedResNet, self).__init__()
        
        # Load a pre-trained ResNet model
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 1 input channel
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Freeze the layers for fine-tuning
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Define the classification head
        self.classification_head = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)  # Output: num_classes for classification
        )
        
        # Define the bounding box regression head
        self.bbox_regression_head = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # Output: 2 for bounding box regression (center, width)
        )
        
        # Replace the fully connected layer of the base model
        self.base_model.fc = nn.Identity()  # Remove the original fully connected layer

        # Load pre-trained weights if provided
        if pretrained_weights_path:
            self.load_pretrained_weights(pretrained_weights_path)

    def forward(self, x):
        features = self.base_model(x)
        class_output = self.classification_head(features)
        bbox_output = self.bbox_regression_head(features)
        return torch.cat((class_output, bbox_output), dim=1)  # Combine outputs

    def load_pretrained_weights(self, path):
        """Load pre-trained weights and freeze layers."""
        pretrained_dict = torch.load(path)
        self.base_model.load_state_dict(pretrained_dict, strict=False)
        
        # Optionally freeze the layers after loading weights
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, layer_names):
        """Unfreeze specific layers for fine-tuning."""
        for name, param in self.base_model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = True