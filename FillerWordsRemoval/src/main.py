import torch
import torch.nn as nn
import torchvision

class TwoHeadedResNet(nn.Module):
    def __init__(self, num_classes, pretrained_weights_path=None):
        super(TwoHeadedResNet, self).__init__()
        # Load the ResNet model
        self.base_model = torchvision.models.resnet18(weights=None)
        
        # Modify the first convolutional layer to accept 1 input channel
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
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
        return torch.cat((class_output, bbox_output), dim=1)  # Concatenate outputs

    def load_pretrained_weights(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.base_model.state_dict()
        
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # Update the model's state_dict
        model_dict.update(pretrained_dict)
        self.base_model.load_state_dict(model_dict)

    def freeze_layers(self):
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze all layers of the base model

        for param in self.classification_head.parameters():
            param.requires_grad = True  # Unfreeze classification head

        for param in self.bbox_regression_head.parameters():
            param.requires_grad = True  # Unfreeze bounding box regression head