"""
Lightweight CNN model for galaxy morphology classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightGalaxyCNN(nn.Module):
    """
    A lightweight CNN for galaxy morphology classification.
    Designed to be efficient while maintaining good accuracy.
    """
    
    def __init__(self, num_classes=3, dropout=0.5):
        super(LightweightGalaxyCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EfficientGalaxyNet(nn.Module):
    """
    An even more efficient model using depthwise separable convolutions.
    Inspired by MobileNet architecture.
    """
    
    def __init__(self, num_classes=3, dropout=0.5):
        super(EfficientGalaxyNet, self).__init__()
        
        def depthwise_separable_conv(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                         padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            depthwise_separable_conv(32, 64, stride=1),
            depthwise_separable_conv(64, 128, stride=2),
            depthwise_separable_conv(128, 128, stride=1),
            depthwise_separable_conv(128, 256, stride=2),
            depthwise_separable_conv(256, 256, stride=1),
            depthwise_separable_conv(256, 512, stride=2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_model(model_name='lightweight', num_classes=3, pretrained=False):
    """
    Get a model instance.
    
    Args:
        model_name: 'lightweight' or 'efficient'
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (not implemented for custom models)
    """
    if model_name == 'lightweight':
        model = LightweightGalaxyCNN(num_classes=num_classes)
    elif model_name == 'efficient':
        model = EfficientGalaxyNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the models
    print("Testing LightweightGalaxyCNN:")
    model1 = LightweightGalaxyCNN(num_classes=3)
    x = torch.randn(1, 3, 224, 224)
    out = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model1):,}")
    
    print("\nTesting EfficientGalaxyNet:")
    model2 = EfficientGalaxyNet(num_classes=3)
    out = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {count_parameters(model2):,}")

