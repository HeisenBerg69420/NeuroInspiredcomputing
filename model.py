"""CNN architecture shared by training and export.

This is a new baseline: the original model.py contained no architecture.
"""

from torch import nn


class ProkrastCNN(nn.Module):
    def __init__(self, number_of_classes: int):
        super().__init__()
        if number_of_classes < 1:
            raise ValueError("number_of_classes must be positive")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, number_of_classes)
        )

    def forward(self, images):
        return self.classifier(self.features(images))
